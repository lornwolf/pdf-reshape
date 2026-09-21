#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""核心处理（pdf_reshape.py、pdf_reshape_store.py）的回归测试，不需要界面。

用法:  python tests/test_core.py [测试名的一部分 ...]
"""
import os
import sqlite3
import sys

import fitz
import numpy as np

from harness import Failures, fixtures, run_tests, work_path
import pdf_reshape as pr
from pdf_reshape_store import Store
from verify import measure


def analyzed(path, **options):
    opts = pr.Options(**options)
    doc = fitz.open(path)
    infos = pr.analyze_document(doc, range(doc.page_count), opts)
    return doc, infos, opts


def process(name, out_name, ref_extra=None, **kwargs):
    """分析并处理一个合成测试文件，返回 (输出路径, infos, ref)。"""
    doc, infos, opts = analyzed(fixtures()[name])
    ref = pr.compute_reference(infos)
    ref.update(ref_extra or {})
    out = work_path(out_name)
    pr.process_document(doc, infos, opts, out, ref=ref, **kwargs)
    return out, infos, ref


def page_array(path, index, dpi=50):
    pix = fitz.open(path)[index].get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)


# ---------------------------------------------------------------- 纠偏与居中

def test_horizontal():
    f = Failures()
    out, infos, _ = process("h.pdf", "core_h_out.pdf")
    from make_test import HORIZONTAL
    for info, spec in zip(infos, HORIZONTAL):
        f.close(info.angle, -spec[1], 0.11, f"第 {info.index + 1} 页检测到的倾斜")
    pages = measure(out)
    f.check(len(pages) == 8, f"输出应为 8 页，实际 {len(pages)}")
    for i, m in enumerate(pages, 1):
        f.close(m["angle"], 0, 0.11, f"第 {i} 页的残余倾斜")
        f.close(m["left"], m["right"], 0.02, f"第 {i} 页左右边距")
        if i != 6:
            f.close(m["top"], m["bottom"], 0.01, f"第 {i} 页上下边距")
    # 第 6 页是章末半页：上边要和别的页对齐，不能被垂直居中到页面中间
    f.close(pages[5]["top"], pages[2]["top"], 0.01, "半页的上边距应与整页一致")
    f.check(pages[5]["bottom"] > 0.5, "半页的下边距应该很大（没有被垂直居中）")
    return f


def test_vertical():
    f = Failures()
    out, infos, ref = process("v.pdf", "core_v_out.pdf")
    f.check(ref["vertical"] is True, "竖排的书应判定为竖排")
    for i, m in enumerate(measure(out), 1):
        f.close(m["angle"], 0, 0.11, f"第 {i} 页的残余倾斜")
        f.close(m["left"], m["right"], 0.01, f"第 {i} 页左右边距")
        f.close(m["top"], m["bottom"], 0.01, f"第 {i} 页上下边距")
    return f


def test_image_mask():
    """图像蒙版（没有色彩空间的 1bit 图）：曾经报 source colorspace must not be None。"""
    f = Failures()
    out, infos, _ = process("mask.pdf", "core_mask_out.pdf")
    f.check(all(p.mode == "native" and p.bilevel for p in infos), "蒙版页应直接取用内嵌原图，并识别为黑白二值")
    image = fitz.open(out)[0].get_images(full=True)[0]
    f.check((image[2], image[3], image[4]) == (1240, 1754, 1), f"输出应保持原分辨率的 1bit 图，实际 {image[2:5]}")
    f.check((page_array(out, 1) > 128).mean() > 0.85, "输出的黑白不应反相")
    for i, m in enumerate(measure(out), 1):
        f.close(m["angle"], 0, 0.11, f"第 {i} 页的残余倾斜")
    return f


def test_one_bad_page_does_not_kill_the_book():
    f = Failures()
    original = pr.analyze_page

    def flaky(doc, index, opts):
        if index == 2:
            raise ValueError("模拟的读取错误")
        return original(doc, index, opts)

    pr.analyze_page = flaky
    try:
        doc, infos, opts = analyzed(fixtures()["h.pdf"])
    finally:
        pr.analyze_page = original
    f.check(infos[2].mode == "copy" and "模拟的读取错误" in infos[2].note, "出错的页应记为原样保留并写明原因")
    out = work_path("core_flaky_out.pdf")
    pr.process_document(doc, infos, opts, out)
    f.check(fitz.open(out).page_count == 8, "其余页应照常输出")
    return f


# ---------------------------------------------------------------- 逐页设定

def test_skip_delete_cleanup():
    f = Failures()
    out, infos, _ = process("h.pdf", "core_marks_out.pdf", ref_extra={"cleanup": {0}},
                            skip_pages={3}, delete_pages={1})
    pages = measure(out)
    f.check(len(pages) == 7, f"删除 1 页后应输出 7 页，实际 {len(pages)}")
    f.close(abs(pages[2]["angle"]), 3.8, 0.11, "不修正的页（原第 4 页）应保持原来的倾斜")
    f.close(pages[1]["angle"], 0, 0.11, "其余页照常纠偏")
    cleaned, untouched_edge = page_array(out, 0), page_array(out, 5)      # 原第 1 页去污；原第 7 页有黑边但没去污
    f.check(cleaned[:8].min() > 200 and cleaned[:, :8].min() > 200, "去除边缘污染后，页边不应再有黑边")
    f.check(min(untouched_edge[:8].min(), untouched_edge[:, :8].min(), untouched_edge[-8:].min()) < 100,
            "没有指定去污的页应保留原样（黑边还在）")
    return f


def test_remap_toc():
    f = Failures()
    toc = [[1, "封面", 1], [1, "第一章", 3], [2, "1.1", 4], [1, "第二章", 7], [1, "附录", 8]]
    got = pr.remap_toc(toc, [0, 2, 3, 4, 5])            # 删掉原第 2、7、8 页
    f.check([e[2] for e in got] == [1, 2, 3, 5, 5], f"删页后书签的页码换算不对：{[e[2] for e in got]}")
    return f


def test_box_adjust_and_neighbor():
    f = Failures()
    boxes = {0: (0.10, 0.1, 0.85, 0.9), 1: (0.17, 0.1, 0.92, 0.9)}           # 左右页的版心位置差 7%
    infos = [pr.PageInfo(index=i, mode="native", bbox=boxes[i % 2]) for i in range(8)]
    infos[4].bbox = (0.10, 0.1, 0.60, 0.5)                                    # 这一页的红框判断得不好
    ref = pr.compute_reference(infos)
    ref["adjust"] = {4: pr.box_like_neighbor(infos[4], infos[3], ref)}        # 与前页（另一侧的页）相同
    for got, want in zip(pr.page_box(infos[4], ref), boxes[0]):
        f.close(got, want, 0.002, "照搬邻页红框后的坐标")
    ref["adjust"] = {2: (-0.02, 0.0, 0.0, 0.0)}                               # 手动把左边框向左扩 2%
    f.close(pr.page_box(infos[2], ref)[0], 0.08, 1e-6, "手动微调后的左边")
    f.check(pr.page_box(infos[6], ref) == boxes[0], "微调只影响那一页")
    return f


# ---------------------------------------------------------------- 版心与对齐规则

def test_dense_extent_ignores_hanging_page_number():
    f = Failures()
    ink = np.zeros((1200, 900), np.uint8)
    for k in range(30):
        ink[100 + 30 * k:116 + 30 * k, 200:800] = 255      # 30 行正文
    ink[1050:1080, 60:120] = 255                           # 挂在正文外侧的页码：实心的一块，墨迹不比正文少
    lo, hi = pr.dense_extent(ink)
    f.close(lo, 200 / 900, 0.005, "正文的左边（不应被页码拉到 60）")
    f.close(hi, 800 / 900, 0.005, "正文的右边")
    f.check(pr.dense_extent(ink.T) is None, "横排的页在竖直方向上分不出「行」，应返回 None 而不是报错")
    return f


def test_peel_margin_stain():
    f = Failures()
    normal = (0.15, 0.1, 0.9, 0.9)
    infos = [pr.PageInfo(index=i, mode="native", bbox=normal, core_bbox=normal, outliers=[]) for i in range(6)]
    infos[3].bbox = (0.05, 0.1, 0.9, 0.9)                  # 左边空白处有一个小污点，把版心撑大了 10%
    infos[3].outliers = [(0.05, 0.5, 0.06, 0.52)]
    infos[4].bbox = (0.15, 0.1, 0.9, 0.95)                 # 带页码的页：比核心版心高，但和别的页一样大
    infos[4].outliers = [(0.5, 0.93, 0.55, 0.95)]
    ref = pr.compute_reference(infos)
    f.close(ref["boxes"][3][0], 0.15, 1e-6, "被污点撑大的一边应被剥掉")
    return f


def test_axis_shift_rules():
    f = Failures()
    edges = [(0.10, 0.85), (0.17, 0.92)]                   # 本页所属的一类、另一类
    f.close(pr.axis_shift(0.20, 0.95, 0.75, edges, True), -0.075, 1e-6, "整页：单独居中")
    f.close(pr.axis_shift(0.15, 0.85, 0.75, edges, True), 0.025, 1e-6, "左侧缩进、右缘对齐：贴齐右边")
    f.close(pr.axis_shift(0.22, 0.92, 0.75, edges, True), -0.045, 1e-6, "页序的奇偶反了：按另一类贴齐")
    f.close(pr.axis_shift(0.125, 0.825, 0.75, edges, True), (1 - 0.7) / 2 - 0.125, 1e-6, "两边缩得差不多：居中")
    f.close(pr.axis_shift(0.10, 0.40, 0.80, [(0.10, 0.90)] * 2, False), 0.0, 1e-6, "章末半页：贴齐上边")
    return f


def test_jpeg_block_aligned_shift():
    """原图是 JPEG 时，不旋转的页的平移量要凑成 8 像素的整数倍（还要算上原图在画布里的偏移）。"""
    f = Failures()
    img = np.zeros((400, 400), np.uint8)
    img[200, 200] = 255
    info = pr.PageInfo(index=0, jpeg_block=8, grid=(3, 5))
    out = pr.transform_image(img, info, 0.0, (0.0123, -0.0177), False)
    y, x = np.argwhere(out == 255)[0]
    f.check((x - 200 + 3) % 8 == 0 and (y - 200 + 5) % 8 == 0, f"平移量 ({x - 200}, {y - 200}) 没有对齐到编码块")
    return f


# ---------------------------------------------------------------- 建议值与大小预估

def test_recommendations():
    f = Failures()
    doc, infos, opts = analyzed(fixtures()["h.pdf"])
    rec = pr.format_recommendation(infos, pr.source_profile(doc, infos), opts.max_angle)
    f.check(rec["min_angle"] == 0.2, f"JPEG 扫描的不旋转角度应建议 0.2，实际 {rec['min_angle']}")
    f.check(rec["quality"] == 95, f"原图质量约 92，应建议 95，实际 {rec['quality']}")
    f.check(rec["max_angle"] == 5.0 and rec["dpi"] is None, "检测范围够用、没有需要渲染的页时，这两项不给新的建议")

    doc, infos, opts = analyzed(fixtures()["mask.pdf"])
    rec = pr.format_recommendation(infos, pr.source_profile(doc, infos), opts.max_angle)
    f.check(rec["min_angle"] == 0.3 and rec["quality"] is None, "150 DPI 的黑白页：角度建议 0.3，JPEG 质量不起作用")

    doc, infos, opts = analyzed(fixtures()["wide.pdf"])
    rec = pr.format_recommendation(infos, pr.source_profile(doc, infos), opts.max_angle)
    f.check(rec["max_angle"] == 8.0, f"有页顶在 ±5° 的边上，应建议放宽到 8，实际 {rec['max_angle']}")
    f.check(rec["dpi"] == 150, f"需要渲染的页里图像是 150 DPI，应建议 150，实际 {rec['dpi']}")
    doc, infos, opts = analyzed(fixtures()["wide.pdf"], max_angle=8.0)
    f.close(abs(infos[1].angle), 6.5, 0.11, "放宽范围后第 2 页的倾斜")
    f.close(abs(infos[3].angle), 7.2, 0.11, "放宽范围后第 4 页的倾斜")
    return f


def test_estimate_output_size():
    f = Failures()
    for quality in (90, 60):
        doc, infos, opts = analyzed(fixtures()["h.pdf"], quality=quality)
        ref = pr.compute_reference(infos)
        estimate = pr.estimate_output_size(doc, infos, ref, opts)[0]
        out = work_path(f"core_estimate_{quality}.pdf")
        pr.process_document(doc, infos, opts, out, ref=ref)
        actual = os.path.getsize(out)
        f.check(abs(estimate - actual) <= 0.1 * actual, f"质量 {quality}：预估 {estimate} 与实际 {actual} 相差超过 10%")
    return f


# ---------------------------------------------------------------- 进度数据库

def fresh_db(name):
    path = work_path(name)
    if os.path.exists(path):
        os.remove(path)
    return path


def test_store_roundtrip():
    f = Failures()
    doc, infos, opts = analyzed(fixtures()["h.pdf"])
    store = Store(fresh_db("core_store.db"))
    book, created = store.open_book(fixtures()["h.pdf"], 8)
    f.check(created, "第一次打开应新建记录")
    store.save_book(book["id"], options={"quality": 70}, output_path="x.pdf", page_range="1-3", current_page=5)
    store.set_page_flag(book["id"], 3, "skip", True)
    store.set_page_flag(book["id"], 1, "deleted", True)
    store.set_page_flag(book["id"], 0, "cleanup", True)
    store.set_box_adjust(book["id"], 2, (-0.02, 0, 0, 0.01))
    store.save_analysis(book["id"], infos, (opts.max_angle, opts.dpi))
    store.close()

    store = Store(work_path("core_store.db"))
    renamed = work_path("core_store_renamed.pdf")
    with open(fixtures()["h.pdf"], "rb") as src, open(renamed, "wb") as dst:
        dst.write(src.read())
    book, created = store.open_book(renamed, 8)             # 同一个文件换了名字：按内容认得出来
    f.check(not created, "改名后的文件应按内容指纹匹配到原来的记录")
    f.check(Store.options_of(book) == {"quality": 70} and book["page_range"] == "1-3" and book["current_page"] == 5,
            "选项、处理范围、当前页应原样恢复")
    f.check(store.flagged_pages(book["id"], "skip") == {3} and store.flagged_pages(book["id"], "deleted") == {1}
            and store.flagged_pages(book["id"], "cleanup") == {0}, "逐页的开关应原样恢复")
    f.check(store.box_adjusts(book["id"]) == {2: (-0.02, 0, 0, 0.01)}, "版心微调应原样恢复")
    restored = store.load_analysis(book, (opts.max_angle, opts.dpi))
    same = lambda a, b: (a is None and b is None) or np.allclose(np.array(a, float).ravel(), np.array(b, float).ravel())
    f.check(restored is not None and len(restored) == len(infos), "分析结果应能取回")
    for a, b in zip(restored or [], infos):             # JSON 里元组会变成列表，所以按数值比
        f.check(same(a.bbox, b.bbox) and same(a.core_bbox, b.core_bbox) and same(a.dense_bbox, b.dense_bbox)
                and same(a.outliers or None, b.outliers or None) and a.angle == b.angle and a.mode == b.mode,
                f"第 {a.index + 1} 页的分析结果没有原样恢复")
    f.check(store.load_analysis(book, (8.0, opts.dpi)) is None, "分析参数变了，旧的分析结果不能复用")
    store.db.execute("UPDATE books SET analysis_version = ?", (pr.ANALYSIS_VERSION - 1,))
    store.db.commit()
    book, _ = store.open_book(renamed, 8)
    f.check(store.load_analysis(book, (opts.max_angle, opts.dpi)) is None, "算法版本变了，旧的分析结果不能复用")
    store.close()
    return f


def test_store_migrates_old_schema():
    f = Failures()
    path = fresh_db("core_old_schema.db")
    db = sqlite3.connect(path)                              # 第一版的表结构：pages 只有 skip 和 analysis
    db.executescript("""
        CREATE TABLE books (id INTEGER PRIMARY KEY, fingerprint TEXT NOT NULL UNIQUE, path TEXT NOT NULL,
            page_count INTEGER NOT NULL, output_path TEXT, options TEXT, page_range TEXT, current_page INTEGER DEFAULT 1,
            analysis_version INTEGER, analysis_params TEXT, processed_at TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE pages (book_id INTEGER NOT NULL, page_index INTEGER NOT NULL, skip INTEGER NOT NULL DEFAULT 0,
            analysis TEXT, PRIMARY KEY (book_id, page_index));
        INSERT INTO books (fingerprint, path, page_count, created_at, updated_at) VALUES ('f', 'x.pdf', 3, 't', 't');
        INSERT INTO pages (book_id, page_index, skip) VALUES (1, 2, 1);""")
    db.commit()
    db.close()
    store = Store(path)
    columns = {r["name"] for r in store.db.execute("PRAGMA table_info(pages)")}
    f.check({"deleted", "cleanup", "box_adjust"} <= columns, f"旧数据库应自动补上新增的列，实际 {sorted(columns)}")
    f.check(store.flagged_pages(1, "skip") == {2}, "旧数据应保留")
    store.set_page_flag(1, 0, "cleanup", True)
    f.check(store.flagged_pages(1, "cleanup") == {0}, "升级后新的列应可以读写")
    store.close()
    return f


def test_store_prune_missing():
    f = Failures()
    store = Store(fresh_db("core_prune.db"))
    kept = fixtures()["h.pdf"]
    gone = work_path("core_prune_gone.pdf")
    with open(fixtures()["v.pdf"], "rb") as src, open(gone, "wb") as dst:
        dst.write(src.read())
    store.open_book(kept, 8)
    book, _ = store.open_book(gone, 4)
    store.set_page_flag(book["id"], 1, "skip", True)
    store.db.execute("INSERT INTO books (fingerprint, path, page_count, created_at, updated_at) "
                     "VALUES ('q', 'Q:\\\\不存在的盘\\\\书.pdf', 5, 't', 't')")
    store.db.commit()
    os.remove(gone)
    removed = store.prune_missing()
    f.check(removed == [gone], f"只应删掉文件已经不在的那一条，实际 {removed}")
    left = sorted(os.path.basename(r["path"]) for r in store.db.execute("SELECT path FROM books"))
    f.check(left == ["h.pdf", "书.pdf"], f"文件还在的、所在的盘访问不到的，都应保留，实际 {left}")
    f.check(store.db.execute("SELECT COUNT(*) FROM pages WHERE book_id = ?", (book["id"],)).fetchone()[0] == 0,
            "被删的书的逐页设定应一并清掉")
    store.close()
    return f


TESTS = [(name[5:], fn) for name, fn in list(globals().items()) if name.startswith("test_") and callable(fn)]

if __name__ == "__main__":
    wanted = sys.argv[1:]
    selected = [(n, fn) for n, fn in TESTS if not wanted or any(w in n for w in wanted)]
    sys.exit(1 if run_tests(selected) else 0)
