#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""图形界面的回归测试：自动驱动界面点按钮、翻页、处理输出，再检查结果。

用法:
    python tests/test_gui.py                # 依次运行全部场景（每个场景一个子进程，带超时）
    python tests/test_gui.py marks keys     # 只运行名字里含这些词的场景
    python tests/test_gui.py --run marks    # 在当前进程里运行一个场景（调试用，会弹出界面）

运行时会弹出程序窗口并自己操作，期间不要动鼠标键盘去碰它。
所有数据库和输出文件都在系统临时目录下，不会碰用户真实的校正进度。
"""
import os
import shutil
import sqlite3
import subprocess
import sys

import fitz
import numpy as np

from harness import fixtures, work_path
from verify import measure


def page_array(path, index, dpi=50):
    pix = fitz.open(path)[index].get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)


# ---------------------------------------------------------------- 场景

def basic(d):
    """打开 → 后台分析 → 预览 → 处理输出。"""
    d.open(fixtures()["h.pdf"])
    d.check(d.busy(), "打开文件后应立即在后台开始分析全书")
    d.check(d.app.var_output.get().endswith("h（校正版）.pdf"), "输出文件名默认是「原文件名（校正版）.pdf」")
    yield d.idle
    d.check(not d.app.btn_reanalyze.winfo_manager(), "分析是现做的，不应出现「重新分析」按钮")
    d.goto(4)
    yield d.previewed
    d.check("旋转 -3.80°" in d.status(), f"第 4 页的状态应显示旋转 -3.80°，实际: {d.status()}")
    d.app.var_clean.set(True)
    out = d.process_to("gui_basic_out.pdf")
    yield d.processed
    d.check("完成" in d.log().splitlines()[-1] and "实际大小" in d.log().splitlines()[-1], "日志末尾应写明完成和实际大小")
    pages = measure(out)
    d.check(len(pages) == 8, "应输出 8 页")
    for i, m in enumerate(pages, 1):
        d.failures.close(m["angle"], 0, 0.11, f"第 {i} 页的残余倾斜")
        d.failures.close(m["left"], m["right"], 0.02, f"第 {i} 页左右边距")


def keys(d):
    """键盘翻页；焦点在数值框里时方向键不翻页。"""
    d.open(fixtures()["h.pdf"])
    yield d.idle
    canvas = d.app.canvases[0]
    d.app.var_page.set(3)
    for key, want in (("<Down>", 4), ("<Right>", 5), ("<Up>", 4), ("<Left>", 3), ("<Next>", 4), ("<Prior>", 3),
                      ("<End>", 8), ("<Down>", 8), ("<Home>", 1), ("<Up>", 1)):
        d.press(canvas, key)
        d.check(d.app.var_page.get() == want, f"在预览区按 {key} 应到第 {want} 页，实际第 {d.app.var_page.get()} 页")
    spin = d.find_widget("TSpinbox", d.app.var_min_angle)
    d.app.var_page.set(3)
    before = d.app.var_min_angle.get()
    d.press(spin, "<Up>")
    d.check(d.app.var_page.get() == 3 and d.app.var_min_angle.get() > before, "焦点在数值框里时，上下键应改数值、不翻页")
    d.press(spin, "<Next>")
    d.check(d.app.var_page.get() == 4, "PageDown 在数值框里也应翻页")


def marks(d):
    """逐页的手动设定：不修正、删除、去除边缘污染、微调版心、与前页相同；处理输出。"""
    app = d.app
    d.open(fixtures()["h.pdf"])
    yield d.idle

    d.goto(4)
    d.click(app.btn_skip)
    d.check(app.btn_skip.cget("text") == "恢复本页修正" and app.skipped == {3}, "「本页不修正」按下后应改名并记下这一页")
    d.goto(5)
    d.check(app.btn_skip.cget("text") == "本页不修正", "翻到别的页，按钮应回到那一页的状态")

    d.goto(2)
    d.click(app.btn_delete)
    d.check(app.btn_delete.cget("text") == "恢复当前页" and app.deleted == {1}, "「删除当前页」按下后应改名为「恢复当前页」")
    yield d.previewed
    d.check("已删除" in d.status(), f"删除的页状态栏应注明，实际: {d.status()}")

    d.goto(1)
    d.check(str(app.btn_like_prev.cget("state")) == "disabled", "第一页的「与前页相同」应不可用")
    d.click(app.btn_cleanup)
    d.check(app.btn_cleanup.cget("text") == "取消去除污染" and app.cleanup == {0}, "「去除边缘污染」按下后应改名")
    yield d.previewed
    d.check("去除边缘污染" in d.status(), f"状态栏应注明去除边缘污染，实际: {d.status()}")
    before, after = app.preview_data[0], app.preview_data[1]
    d.check(before[:30, :30].min() < 60 and after[:30, :30].min() > 200, "预览里黑边应在处理后消失")

    d.goto(3)                                                   # 手动微调版心边框
    arrows = lambda: "".join(a for a, b in app.nudge_buttons.items() if str(b.cget("state")) == "normal")
    app.var_edge.set("上"); app.update_nudge_buttons()
    d.check(arrows() == "↑↓", f"选「上」时只有上下箭头可用，实际 {arrows()}")
    app.var_edge.set("左"); app.update_nudge_buttons()
    d.check(arrows() == "←→", f"选「左」时只有左右箭头可用，实际 {arrows()}")
    yield d.previewed
    old_status = d.status()
    for _ in range(10):
        d.click(app.nudge_buttons["←"])
    d.check(app.adjust == {2: (-0.02, 0.0, 0.0, 0.0)}, f"左边框左移 10 步应为 -2%，实际 {app.adjust}")
    yield d.previewed
    d.check(d.status() != old_status and "x-2.8%" in d.status(), f"微调后应重新计算对齐（x-3.8% → x-2.8%），实际: {d.status()}")

    d.goto(6)                                                   # 半页的红框照前页的来
    d.click(app.btn_like_prev)
    d.check(d.frame_after(5) == d.frame_after(4), f"「与前页相同」后红框应与前页一致: {d.frame_after(5)} / {d.frame_after(4)}")
    d.check("去污 1 页: 1" in app.lbl_skipped.cget("text") and "删除 1 页: 2" in app.lbl_skipped.cget("text"),
            f"标记列表不对: {app.lbl_skipped.cget('text')}")

    out = d.process_to("gui_marks_out.pdf")
    yield d.processed
    pages = measure(out)
    d.check(len(pages) == 7, f"删除 1 页后应输出 7 页，实际 {len(pages)}")
    d.failures.close(abs(pages[2]["angle"]), 3.8, 0.11, "不修正的页应保持原来的倾斜")
    edge = page_array(out, 0)
    d.check(edge[:8].min() > 200 and edge[:, :8].min() > 200, "去污的页输出后不应再有黑边")
    d.check(any("删除 1 页" in m[2] for m in d.messages), "完成提示里应写明删除了几页")


def marks_restore(d):
    """接着上一个场景：关掉再打开，设定都应恢复；复位、取消、重新分析。"""
    app = d.app
    d.open(fixtures()["h.pdf"])
    d.check(app.infos is not None and not d.busy(), "有保存的分析结果时应直接恢复，不重新分析")
    d.check(app.skipped == {3} and app.deleted == {1} and app.cleanup == {0} and set(app.adjust) == {2, 5},
            f"逐页设定没有恢复: {app.skipped} {app.deleted} {app.cleanup} {sorted(app.adjust)}")
    d.check(bool(app.btn_reanalyze.winfo_manager()), "分析结果是从历史记录恢复的，应出现「重新分析」按钮")
    d.check(app.var_page.get() == 6, f"应回到上次看的页，实际第 {app.var_page.get()} 页")
    d.goto(3)
    d.click(app.btn_nudge_reset)
    d.check(set(app.adjust) == {5} and str(app.btn_nudge_reset.cget("state")) == "disabled", "「复位」应撤销这一页的微调")
    d.goto(2)
    d.click(app.btn_delete)
    d.check(app.deleted == set() and app.btn_delete.cget("text") == "删除当前页", "「恢复当前页」应取消删除")
    d.click(app.btn_reanalyze)
    d.check(d.busy() and app.infos is None, "「重新分析」应丢掉恢复来的结果并开始分析")
    yield d.idle
    d.check(app.skipped == {3} and app.cleanup == {0}, "重新分析不应丢掉手动设定")


def recommend(d):
    """四项参数的建议值；一个按钮在「采用建议值」和「恢复默认值」之间切换；分析参数变了自动重新分析。"""
    app = d.app
    d.open(fixtures()["wide.pdf"])
    yield d.idle
    d.check(app.rec == {"max_angle": 8.0, "min_angle": 0.2, "quality": 90, "dpi": 150}, f"建议值不对: {app.rec}")
    shown = {n for n, label in app.rec_labels.items() if label.winfo_manager()}
    d.check(shown == {"max_angle", "min_angle", "quality", "dpi"}, f"四项都有事可做，都应显示提示，实际 {shown}")
    d.check(app.btn_recommend.cget("text") == "采用建议值", "有一项不是建议值时按钮应为「采用建议值」")
    d.click(app.btn_recommend)
    d.check(d.options()["max_angle"] == 8.0 and d.options()["dpi"] == 150 and d.options()["min_angle"] == 0.2,
            f"应一次采用全部建议值，实际 {d.options()}")
    d.check(d.busy(), "倾斜检测范围、渲染 DPI 变了应立即自动重新分析")
    yield d.idle
    d.failures.close(abs(app.infos[1].angle), 6.5, 0.11, "放宽范围后第 2 页的倾斜")
    d.failures.close(abs(app.infos[3].angle), 7.2, 0.11, "放宽范围后第 4 页的倾斜")
    d.check(app.btn_recommend.cget("text") == "恢复默认值", "全部采用后按钮应改名为「恢复默认值」")
    app.var_quality.set(80)                                     # 手动改掉一项
    d.refresh()
    d.check(app.btn_recommend.cget("text") == "采用建议值", "手动改了数值，按钮应变回「采用建议值」")
    d.click(app.btn_recommend)
    d.check(d.options()["quality"] == 90 and not d.busy(), "只是质量变了，不需要重新分析")
    d.click(app.btn_recommend)                                  # 恢复默认值
    d.check(d.options()["max_angle"] == 5.0 and d.options()["min_angle"] == 0.1 and d.options()["dpi"] == 300,
            f"应一次恢复全部默认值，实际 {d.options()}")
    yield d.idle


def recommend_quiet(d):
    """普通的书：倾斜检测范围和渲染 DPI 没事可做，不占提示行。"""
    d.open(fixtures()["h.pdf"])
    yield d.idle
    shown = {n for n, label in d.app.rec_labels.items() if label.winfo_manager()}
    d.check(shown == {"min_angle", "quality"}, f"只应显示角度和质量两项，实际 {shown}")
    d.click(d.app.btn_recommend)
    d.check(not d.busy(), "采用角度和质量的建议值不需要重新分析")


def estimate(d):
    """输出大小的预估：设置一变就重算，翻页不重算，和实际大小接近。"""
    app = d.app
    d.open(fixtures()["h.pdf"])
    yield d.estimated
    first = app.lbl_estimate.cget("text")
    app.var_quality.set(60)
    d.refresh()
    yield 1200
    yield d.estimated
    second = app.lbl_estimate.cget("text")
    d.check(second != first, f"JPEG 质量变了应重新预估: {first} → {second}")
    count = app.estimate_gen
    d.goto(5)
    yield 1500
    d.check(app.estimate_gen == count, "只是翻页，不应重新预估")
    out = d.process_to("gui_estimate_out.pdf")
    yield d.processed
    estimated = float(second.split("约")[1].split("MB")[0].split("KB")[0]) * (1048576 if "MB" in second.split("（")[0] else 1024)
    actual = os.path.getsize(out)
    d.check(abs(estimated - actual) <= 0.1 * actual, f"预估 {estimated:.0f} 与实际 {actual} 相差超过 10%")


def estimate_debounce(d):
    """连着点版心边框的箭头时不重新预估大小，停手 2 秒之后才算一次。"""
    import time
    app = d.app
    d.open(fixtures()["h.pdf"])
    yield d.estimated
    d.goto(3)
    yield d.previewed
    starts = []
    original = app.start_estimate

    def spy(opts, indices):
        starts.append(time.monotonic())
        original(opts, indices)
    app.start_estimate = spy
    app.var_edge.set("左")
    app.update_nudge_buttons()
    for _ in range(4):                              # 每隔 1 秒点一下：比普通设置 0.8 秒的延迟长，但不到 2 秒
        d.click(app.nudge_buttons["←"])
        last_click = time.monotonic()
        d.check("调整结束后重新计算" in app.lbl_estimate.cget("text"), "点击后应提示稍后重新计算")
        yield 1000
    d.check(not starts, f"连续点击期间不应重新预估，实际算了 {len(starts)} 次")
    yield lambda: bool(starts) or time.monotonic() - last_click > 6
    d.check(len(starts) == 1, f"停手之后应只预估 1 次，实际 {len(starts)} 次")
    if starts:
        waited = starts[0] - last_click
        d.check(1.9 <= waited <= 3.5, f"应在最后一次点击约 2 秒后开始预估，实际 {waited:.1f} 秒")
    yield d.estimated
    d.check(app.lbl_estimate.cget("text").startswith("预计输出约"), "算完后应显示新的预估值")
    starts.clear()
    app.var_quality.set(60)                         # 别的设置变了，还是按 0.8 秒的延迟
    changed = time.monotonic()
    d.refresh()
    yield lambda: bool(starts) or time.monotonic() - changed > 6
    d.check(bool(starts) and starts[0] - changed < 1.8, "JPEG 质量变了应按普通的延迟（0.8 秒）重新预估")
    yield d.estimated


def files(d):
    """每本书的设置互不沿用；有记录的书恢复它自己的设置；改名的文件按内容认得；旧的默认输出名自动更新。"""
    app = d.app
    h, v = fixtures()["h.pdf"], fixtures()["v.pdf"]
    defaults = d.options() | {}
    d.open(h)
    yield d.idle
    app.var_deskew.set(False); app.var_clean.set(True); app.var_upscale.set(True)
    app.var_min_angle.set(0.4); app.var_quality.set(65); app.var_pages.set("2-5")
    app.var_output.set(work_path("自己起的名字.pdf"))
    d.refresh()
    changed = d.options()
    d.open(v)
    d.check(d.options() == defaults and app.var_pages.get() == "", f"打开新文件应恢复默认设置，实际 {d.options()}")
    yield d.idle
    d.open(h)
    d.check(d.options() == changed and app.var_pages.get() == "2-5", f"有记录的书应恢复它自己的设置，实际 {d.options()}")
    d.check(app.var_output.get().endswith("自己起的名字.pdf"), "用户自己起的输出文件名应保留")
    yield d.idle
    renamed = work_path("gui_files_renamed.pdf")
    shutil.copy(h, renamed)
    d.open(renamed)
    d.check(d.options() == changed and not d.busy(), "改名后的文件应按内容匹配到原来的记录，直接恢复")
    yield d.idle
    d.open(v)                                       # 先切到别的书：开着的那本会被自动保存覆盖
    yield d.idle
    # 按文件名匹配：数据库里存的路径是规范化过的（全是反斜杠），和这里拼出来的写法不一定一样
    updated = app.store.db.execute("UPDATE books SET output_path = ? WHERE path LIKE ?",
                                   (os.path.join(os.path.dirname(renamed), "gui_files_renamed_reshaped.pdf"),
                                    "%gui_files_renamed.pdf")).rowcount
    app.store.db.commit()
    d.check(updated == 1, f"测试准备：应改到 1 条记录，实际 {updated}")
    d.open(renamed)
    d.check(app.var_output.get().endswith("gui_files_renamed（校正版）.pdf"),
            f"记录里存的是旧版的默认名 _reshaped，应换成新的默认名，实际 {app.var_output.get()}")


def prepare_stale(db):
    """准备一条带旧版本分析结果的记录。"""
    import pdf_reshape as pr
    from pdf_reshape_store import Store
    opts = pr.Options()
    with fitz.open(fixtures()["v.pdf"]) as doc:
        infos = pr.analyze_document(doc, range(doc.page_count), opts)
    store = Store(db)
    book, _ = store.open_book(fixtures()["v.pdf"], len(infos))
    store.save_analysis(book["id"], infos, (opts.max_angle, opts.dpi))
    store.set_page_flag(book["id"], 1, "skip", True)
    store.db.execute("UPDATE books SET analysis_version = ?", (pr.ANALYSIS_VERSION - 1,))
    store.db.commit()
    store.close()


def stale_analysis(d):
    """检测算法升级后，旧版本留下的分析结果应自动作废、重新分析；手动设定保留。"""
    d.open(fixtures()["v.pdf"])
    d.check(d.app.infos is None and d.busy(), "旧版本的分析结果不应被复用，应重新分析")
    d.check(d.app.skipped == {1}, "手动设定应保留")
    yield d.idle
    version = d.app.store.db.execute("SELECT analysis_version FROM books").fetchone()[0]
    d.check(version == d.core.ANALYSIS_VERSION, f"重新分析后应存成当前的版本号，实际 {version}")


def prepare_prune(db):
    from pdf_reshape_store import Store
    gone = work_path("gui_prune_gone.pdf")
    shutil.copy(fixtures()["v.pdf"], gone)
    store = Store(db)
    store.open_book(fixtures()["h.pdf"], 8)
    store.open_book(gone, 4)
    store.db.execute("INSERT INTO books (fingerprint, path, page_count, created_at, updated_at) "
                     "VALUES ('q', 'Q:\\\\不存在的盘\\\\书.pdf', 5, 't', 't')")
    store.db.commit()
    store.close()
    os.remove(gone)


def prune(d):
    """启动时删掉原文件已经不存在的历史记录；所在的盘访问不到的不删。"""
    d.check("gui_prune_gone.pdf" in d.log() and "删除了它的历史记录" in d.log(), f"日志里应写明删了哪条记录: {d.log()}")
    left = sorted(os.path.basename(r["path"]) for r in d.app.store.list_books())
    d.check(left == ["h.pdf", "书.pdf"], f"剩下的记录不对: {left}")
    win = d.app.show_history()
    d.app.update()
    d.check(win is not None and win.winfo_exists(), "历史记录窗口应能打开")
    win.destroy()
    yield 100


# (名字, 场景, 数据库, 开始前是否清空数据库, 启动界面之前的准备)
SCENARIOS = [
    ("basic", basic, "gui_basic.db", True, None),
    ("keys", keys, "gui_keys.db", True, None),
    ("marks", marks, "gui_marks.db", True, None),
    ("marks_restore", marks_restore, "gui_marks.db", False, None),
    ("recommend", recommend, "gui_recommend.db", True, None),
    ("recommend_quiet", recommend_quiet, "gui_recommend_quiet.db", True, None),
    ("estimate", estimate, "gui_estimate.db", True, None),
    ("estimate_debounce", estimate_debounce, "gui_estimate_debounce.db", True, None),
    ("files", files, "gui_files.db", True, None),
    ("stale_analysis", stale_analysis, "gui_stale.db", True, prepare_stale),
    ("prune", prune, "gui_prune.db", True, prepare_prune),
]
SCENARIO_TIMEOUT = 300


def run_one(name):
    from gui_driver import Driver
    _, scenario, db, fresh, prepare = next(s for s in SCENARIOS if s[0] == name)
    if fresh and os.path.exists(work_path(db)):
        os.remove(work_path(db))
    if prepare:
        prepare(work_path(db))
    failures = Driver(db, timeout=SCENARIO_TIMEOUT).run(scenario)
    for problem in failures:
        print("      " + str(problem).replace("\n", "\n      "))
    return 1 if failures else 0


def run_all(wanted):
    fixtures()                                      # 先把测试文件生成好，各个子进程共用
    names = [s[0] for s in SCENARIOS if not wanted or any(w in s[0] for w in wanted)]
    if "marks_restore" in names and "marks" not in names:
        names.insert(names.index("marks_restore"), "marks")     # 它要接着 marks 留下的数据库
    failed = 0
    for name in names:
        try:
            proc = subprocess.run([sys.executable, os.path.abspath(__file__), "--run", name], text=True,
                                  capture_output=True, encoding="utf-8", errors="replace",
                                  timeout=SCENARIO_TIMEOUT + 60, env=os.environ | {"PYTHONIOENCODING": "utf-8"})
            ok, detail = proc.returncode == 0, (proc.stdout + proc.stderr).rstrip()
        except subprocess.TimeoutExpired:
            ok, detail = False, "      子进程超时，已强行结束"
        print(("ok    " if ok else "FAIL  ") + "gui:" + name)
        if not ok:
            failed += 1
            print(detail)
    return failed


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--run":
        sys.exit(run_one(sys.argv[2]))
    sys.exit(1 if run_all(sys.argv[1:]) else 0)
