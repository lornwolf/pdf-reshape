#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""校正进度的持久化（SQLite）。

每本书一条 books 记录（选项、输出路径、看到第几页……），每页一条 pages 记录
（全书分析的结果 + 用户对这一页的手动调整）。下次打开同一本书时直接恢复，不用重新分析。

书按**文件内容的指纹**识别，不按路径：文件改名、挪位置之后仍然能认出来。
数据库默认在 ~/.pdf_reshape/books.db，可用环境变量 PDF_RESHAPE_DB 指定别的位置。
"""
import hashlib
import json
import os
import sqlite3
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from pdf_reshape import ANALYSIS_VERSION, PageInfo

SCHEMA = """
CREATE TABLE IF NOT EXISTS books (
    id               INTEGER PRIMARY KEY,
    fingerprint      TEXT NOT NULL UNIQUE,   -- 文件内容指纹
    path             TEXT NOT NULL,          -- 最近一次打开时的路径
    page_count       INTEGER NOT NULL,
    output_path      TEXT,
    options          TEXT,                   -- 界面选项（JSON）
    page_range       TEXT,                   -- 「处理范围」输入框的内容
    current_page     INTEGER DEFAULT 1,      -- 上次看到第几页（从 1 开始）
    analysis_version INTEGER,                -- 产生分析结果的算法版本；和当前版本不同就要重新分析
    analysis_params  TEXT,                   -- 产生分析结果时的 [倾斜检测范围, 渲染 DPI]（JSON）
    processed_at     TEXT,                   -- 最近一次输出成功的时间
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS pages (
    book_id     INTEGER NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    page_index  INTEGER NOT NULL,            -- 从 0 开始
    skip        INTEGER NOT NULL DEFAULT 0,  -- 1 = 用户指定「本页不修正」
    deleted     INTEGER NOT NULL DEFAULT 0,  -- 1 = 用户指定「删除当前页」（新 PDF 中不输出）
    cleanup     INTEGER NOT NULL DEFAULT 0,  -- 1 = 用户指定「去除边缘污染」
    box_adjust  TEXT,                        -- 用户对版心四条边的手动微调 [左, 上, 右, 下]（JSON），没调过为 NULL
    analysis    TEXT,                        -- 这一页的分析结果（PageInfo 的 JSON）
    PRIMARY KEY (book_id, page_index)
);
"""


def default_db_path():
    return Path(os.environ.get("PDF_RESHAPE_DB") or Path.home() / ".pdf_reshape" / "books.db")


def fingerprint(path):
    """文件大小 + 首尾各 1MB 的 SHA1。不读整个文件，几百 MB 的扫描书也是瞬间完成。"""
    chunk = 1 << 20
    size = os.path.getsize(path)
    h = hashlib.sha1(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(chunk))
        if size > 2 * chunk:
            f.seek(-chunk, os.SEEK_END)
            h.update(f.read(chunk))
    return h.hexdigest()


def _now():
    return datetime.now().isoformat(timespec="seconds")


def _info_to_json(info):
    return json.dumps(asdict(info), ensure_ascii=False)


def _info_from_json(text):
    d = json.loads(text)
    for key in ("bbox", "core_bbox", "dense_bbox", "scan_rect", "grid"):   # JSON 里元组变成了列表，还原回去
        if d.get(key) is not None:
            d[key] = tuple(d[key])
    known = PageInfo.__dataclass_fields__
    return PageInfo(**{k: v for k, v in d.items() if k in known})


class Store:
    def __init__(self, db_path=None):
        self.path = Path(db_path) if db_path else default_db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(SCHEMA)
        self._migrate()
        self.db.commit()

    def _migrate(self):
        """给旧版本建的数据库补上后来新增的列（CREATE TABLE IF NOT EXISTS 不会改已有的表）。"""
        cols = {r["name"] for r in self.db.execute("PRAGMA table_info(pages)")}
        if "deleted" not in cols:
            self.db.execute("ALTER TABLE pages ADD COLUMN deleted INTEGER NOT NULL DEFAULT 0")
        if "cleanup" not in cols:
            self.db.execute("ALTER TABLE pages ADD COLUMN cleanup INTEGER NOT NULL DEFAULT 0")
        if "box_adjust" not in cols:
            self.db.execute("ALTER TABLE pages ADD COLUMN box_adjust TEXT")

    def close(self):
        self.db.close()

    # ------------------------------------------------------------ 书

    def open_book(self, path, page_count):
        """按文件指纹找到这本书的记录，没有就新建。返回 (记录, 是否新建)。"""
        fp = fingerprint(path)
        row = self.db.execute("SELECT * FROM books WHERE fingerprint = ?", (fp,)).fetchone()
        if row is None:
            now = _now()
            self.db.execute(
                "INSERT INTO books (fingerprint, path, page_count, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (fp, str(path), page_count, now, now))
            created = True
        else:
            self.db.execute("UPDATE books SET path = ?, page_count = ? WHERE id = ?",
                            (str(path), page_count, row["id"]))
            created = False
        self.db.commit()
        return self.db.execute("SELECT * FROM books WHERE fingerprint = ?", (fp,)).fetchone(), created

    def save_book(self, book_id, **fields):
        """更新 books 的若干列。options 传字典，自动转成 JSON。"""
        if "options" in fields:
            fields["options"] = json.dumps(fields["options"], ensure_ascii=False)
        fields["updated_at"] = _now()
        cols = ", ".join(f"{k} = ?" for k in fields)
        self.db.execute(f"UPDATE books SET {cols} WHERE id = ?", (*fields.values(), book_id))
        self.db.commit()

    def list_books(self):
        return self.db.execute("""
            SELECT b.*, (SELECT COUNT(*) FROM pages p WHERE p.book_id = b.id AND p.skip = 1) AS skipped,
                        (SELECT COUNT(*) FROM pages p WHERE p.book_id = b.id AND p.deleted = 1) AS deleted,
                        (SELECT COUNT(*) FROM pages p WHERE p.book_id = b.id AND p.analysis IS NOT NULL) AS analyzed
            FROM books b ORDER BY b.updated_at DESC""").fetchall()

    def delete_book(self, book_id):
        self.db.execute("DELETE FROM books WHERE id = ?", (book_id,))
        self.db.commit()

    def prune_missing(self):
        """删掉原文件已经不存在的书的记录（连同它的逐页设定）。返回被删掉的那些文件的路径。

        只有「所在的盘还在、文件却没了」才算不存在。盘符（或网络路径的根）本身就访问不到——
        移动硬盘没插、网络盘没连上、同步盘没挂载——只是暂时够不着，不能因此把校正进度全删了。
        """
        removed = []
        for book in self.db.execute("SELECT id, path FROM books").fetchall():
            path = Path(book["path"])
            try:
                gone = not path.exists() and bool(path.anchor) and Path(path.anchor).exists()
            except OSError:                 # 路径本身有问题（权限、格式），拿不准就不删
                gone = False
            if gone:
                self.db.execute("DELETE FROM books WHERE id = ?", (book["id"],))
                removed.append(book["path"])
        self.db.commit()
        return removed

    @staticmethod
    def options_of(book):
        return json.loads(book["options"]) if book["options"] else {}

    # ------------------------------------------------------------ 页

    PAGE_FLAGS = ("skip", "deleted", "cleanup")    # pages 表里用户逐页设置的开关

    def set_page_flag(self, book_id, page_index, flag, value):
        assert flag in self.PAGE_FLAGS
        self.db.execute(f"""
            INSERT INTO pages (book_id, page_index, {flag}) VALUES (?, ?, ?)
            ON CONFLICT (book_id, page_index) DO UPDATE SET {flag} = excluded.{flag}""",
                        (book_id, page_index, int(value)))
        self.db.execute("UPDATE books SET updated_at = ? WHERE id = ?", (_now(), book_id))
        self.db.commit()

    def flagged_pages(self, book_id, flag):
        assert flag in self.PAGE_FLAGS
        rows = self.db.execute(f"SELECT page_index FROM pages WHERE book_id = ? AND {flag} = 1", (book_id,))
        return {r["page_index"] for r in rows}

    def set_box_adjust(self, book_id, page_index, offsets):
        """保存某一页版心的手动微调；offsets 为 None 或全 0 表示复位。"""
        value = json.dumps([round(o, 5) for o in offsets]) if offsets and any(offsets) else None
        self.db.execute("""
            INSERT INTO pages (book_id, page_index, box_adjust) VALUES (?, ?, ?)
            ON CONFLICT (book_id, page_index) DO UPDATE SET box_adjust = excluded.box_adjust""",
                        (book_id, page_index, value))
        self.db.execute("UPDATE books SET updated_at = ? WHERE id = ?", (_now(), book_id))
        self.db.commit()

    def box_adjusts(self, book_id):
        rows = self.db.execute(
            "SELECT page_index, box_adjust FROM pages WHERE book_id = ? AND box_adjust IS NOT NULL", (book_id,))
        return {r["page_index"]: tuple(json.loads(r["box_adjust"])) for r in rows}

    def save_analysis(self, book_id, infos, params):
        """保存全书的分析结果。用户的手动调整（skip）原样保留。"""
        self.db.executemany("""
            INSERT INTO pages (book_id, page_index, analysis) VALUES (?, ?, ?)
            ON CONFLICT (book_id, page_index) DO UPDATE SET analysis = excluded.analysis""",
                            [(book_id, p.index, _info_to_json(p)) for p in infos])
        self.save_book(book_id, analysis_version=ANALYSIS_VERSION, analysis_params=json.dumps(list(params)))

    def load_analysis(self, book, params):
        """取回全书的分析结果；算法版本、分析参数变了，或者页数对不上，就返回 None（需要重新分析）。"""
        if book["analysis_version"] != ANALYSIS_VERSION or not book["analysis_params"]:
            return None
        if json.loads(book["analysis_params"]) != list(params):
            return None
        rows = self.db.execute(
            "SELECT analysis FROM pages WHERE book_id = ? AND analysis IS NOT NULL ORDER BY page_index",
            (book["id"],)).fetchall()
        infos = [_info_from_json(r["analysis"]) for r in rows]
        if [p.index for p in infos] != list(range(book["page_count"])):
            return None
        return infos
