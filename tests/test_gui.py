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
    d.open(fixtures()["h.pdf"], analyze=False)
    d.check(not d.busy() and d.app.btn_reanalyze.cget("text") == "分析全书", "打开文件后不应自动分析，等用户点「分析全书」")
    d.check(str(d.app.btn_skip.cget("state")) == "disabled" and str(d.app.btn_delete.cget("state")) == "disabled",
            "分析之前逐页的按钮（含「本页不纠偏居中」「删除当前页」）都不可用")
    d.check(d.app.var_output.get().endswith("h（校正版）.pdf"), "输出文件名默认是「原文件名（校正版）.pdf」")
    d.click(d.app.btn_reanalyze)
    d.check(d.busy(), "点「分析全书」后应在后台开始分析")
    yield d.idle
    d.check(d.app.btn_reanalyze.cget("text") == "重新分析", "分析过之后按钮应变成「重新分析」")
    d.check(str(d.app.btn_skip.cget("state")) == "normal" and str(d.app.btn_delete.cget("state")) == "normal", "分析完之后逐页的按钮可用")
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
    """逐页的手动设定：不纠偏居中、删除、去除边缘污染、微调版心、与前页相同；处理输出。"""
    app = d.app
    d.open(fixtures()["h.pdf"])
    yield d.idle

    d.goto(4)
    d.click(app.btn_skip)
    d.check(app.btn_skip.cget("text") == "恢复纠偏居中" and app.skipped == {3}, "「本页不纠偏居中」按下后应改名并记下这一页")
    d.click(app.btn_cleanup)                                    # 不纠偏居中的页也可以去污（曾经点了没反应）
    yield d.previewed
    d.check("本页不纠偏居中" in d.status() and "去除边缘污染" in d.status(), f"状态栏应同时注明两者，实际: {d.status()}")
    before, after, box = app.preview_data[0], app.preview_data[1], app.preview_data[2]
    d.check(before[:30, :30].min() < 60 and after[:30, :30].min() > 200, "不纠偏居中的页去污后，预览里黑边应消失")
    d.check(box is not None, "不纠偏居中的页也应画出红框（去污是按红框算的）")
    d.click(app.btn_cleanup)                                    # 取消，后面的检查照旧
    d.goto(5)
    d.check(app.btn_skip.cget("text") == "本页不纠偏居中", "翻到别的页，按钮应回到那一页的状态")

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
    d.check(d.status() == old_status, f"只调红框不应移动页面，状态应不变，实际: {d.status()}")
    d.click(app.align_buttons["center"])                        # 按「版心：居中」才按新的红框对齐
    d.check(2 in app.shift, "居中之后应记下这一页的平移量")
    yield d.previewed
    d.check("x-2.8%" in d.status(), f"「居中」后应按新的红框重新居中（x-3.8% → x-2.8%），实际: {d.status()}")
    d.click(app.nudge_buttons["←"])                             # 居中之后再调红框：页面仍然不动
    yield d.previewed
    d.check("x-2.8%" in d.status(), f"再调红框页面不应动，实际: {d.status()}")
    d.click(app.nudge_buttons["→"])

    d.goto(6)                                                   # 半页的红框照前页的来
    d.click(app.btn_like_prev)
    d.click(app.align_buttons["center"])
    d.check(d.frame_after(5) == d.frame_after(4), f"「与前页相同」后红框应与前页一致: {d.frame_after(5)} / {d.frame_after(4)}")
    d.check("去污 1 页: 1" in app.lbl_skipped.cget("text") and "删除 1 页: 2" in app.lbl_skipped.cget("text"),
            f"标记列表不对: {app.lbl_skipped.cget('text')}")

    out = d.process_to("gui_marks_out.pdf")
    yield d.processed
    pages = measure(out)
    d.check(len(pages) == 7, f"删除 1 页后应输出 7 页，实际 {len(pages)}")
    d.failures.close(abs(pages[2]["angle"]), 3.8, 0.11, "不纠偏居中的页应保持原来的倾斜")
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
    d.check(set(app.shift) == {2, 5}, f"指定的平移量没有恢复: {app.shift}")
    d.check(bool(app.btn_reanalyze.winfo_manager()), "分析结果是从历史记录恢复的，应出现「重新分析」按钮")
    d.check(app.var_page.get() == 6, f"应回到上次看的页，实际第 {app.var_page.get()} 页")
    d.goto(3)
    d.click(app.btn_nudge_reset)
    d.check(set(app.adjust) == {5} and set(app.shift) == {5} and str(app.btn_nudge_reset.cget("state")) == "disabled",
            "「复位」应同时撤销这一页的红框微调和指定的位置")
    yield d.previewed
    d.check("x-3.8%" in d.status(), f"复位后应回到自动判断的对齐（x-3.8%），实际: {d.status()}")
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
    d.check(app.rec == {"max_angle": 8.0, "min_angle": 0.2, "quality": 90, "dpi": 150, "flatten": None}, f"建议值不对: {app.rec}")
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


def enhance(d):
    """显示增强：只增强值得增强的页；放大镜；处理输出。"""
    app = d.app
    d.open(fixtures()["mask.pdf"])
    yield d.idle
    d.check("「显示增强」" in d.log() and "平滑放大 ×3 8 页" in d.log(), f"分析完应说明有多少页值得增强: {d.log()[-200:]}")
    d.goto(3)
    yield d.previewed
    d.check("显示增强" not in d.status(), "没勾选时不应增强")
    app.var_enhance.set(True)
    d.refresh()
    yield d.previewed
    d.check("显示增强: 平滑放大 ×3" in d.status(), f"状态栏应注明用了什么方法，实际: {d.status()}")
    before, after, _ = app.preview_full
    d.check(after.shape[1] == 3 * before.shape[1], f"增强后的图应是 3 倍分辨率: {before.shape} → {after.shape}")
    x0, y0, w, h = app.preview_geometry[1]
    app.show_loupe(1, x0 + w // 2, y0 + h // 2)             # 放大镜：两边同时放大同一处
    app.update()
    d.check(all(cv.find_withtag("loupe") for cv in app.canvases), "按住预览图时两边都应出现放大镜")
    app.hide_loupe()
    d.check(not any(cv.find_withtag("loupe") for cv in app.canvases), "松开后放大镜应消失")
    yield d.estimated
    out = d.process_to("gui_enhance_out.pdf")
    yield d.processed
    image = fitz.open(out)[2].get_images(full=True)[0]
    d.check((image[2], image[4]) == (3720, 1), f"输出应是 3 倍分辨率的 1bit 图，实际 {image[2:5]}")


def lock(d):
    """分析和处理期间：会改变输出的控件全部不可用，「取消」、翻页和预览照常；结束或取消之后恢复。"""
    app = d.app
    check_enhance = d.find_widget("TCheckbutton", None, text="显示增强")
    spin_quality = d.find_widget("TSpinbox", app.var_quality)
    entry_output = d.find_widget("TEntry", app.var_output)
    btn_open = d.find_widget("TButton", None, text="打开…")

    def state(w):
        return str(w.cget("state"))

    def editable():
        return [state(w) for w in (check_enhance, spin_quality, entry_output, btn_open, app.btn_skip, app.btn_delete)]

    d.open(fixtures()["h.pdf"])
    d.check(d.busy() and editable() == ["disabled"] * 6, f"分析期间各项设置应不可用，实际 {editable()}")
    d.check(state(app.btn_run) == "disabled" and state(app.btn_cancel) == "normal", "分析期间「开始处理」不可用、「取消」可用")
    d.goto(2)                                                   # 分析期间也能翻页看暂定的预览
    yield d.previewed
    d.check(not d.busy() or editable() == ["disabled"] * 6, "分析期间翻页刷新之后仍应不可用")   # 测试用的书很小，可能已经分析完了
    yield d.idle
    d.check(editable() == ["normal"] * 6 and state(app.btn_run) == "normal", f"分析结束后应恢复，实际 {editable()}")
    d.goto(3)
    yield d.previewed
    d.check(state(app.btn_cleanup) == "normal" and state(app.nudge_buttons["↑"]) == "normal", "分析完之后逐页的调整应可用")

    passed = []
    start_worker = app.start_worker
    app.start_worker = lambda target, *args: (passed.append(args), start_worker(target, *args))
    d.process_to("gui_lock_out.pdf")
    app.start_worker = start_worker
    d.check(d.busy(), "应已开始处理")
    d.check(editable() == ["disabled"] * 6, f"处理期间各项设置应不可用，实际 {editable()}")
    per_page = [app.btn_cleanup, app.btn_like_prev, app.btn_like_next, app.btn_run, *app.nudge_buttons.values()]
    d.check(all(state(w) == "disabled" for w in per_page), "处理期间逐页的调整和「开始处理」应不可用")
    d.check(state(app.btn_cancel) == "normal", "处理期间「取消」必须可用")
    app.btn_skip.invoke()                                       # 不经过 d.click：它会处理事件，小书可能就在这时处理完、解了锁
    app.btn_cleanup.invoke()
    d.check(not app.skipped and not app.cleanup, "处理期间点不可用的按钮不应有任何效果")
    ref = passed[0][2]
    d.check(ref is not app.ref and ref["adjust"] is not app.adjust and ref["cleanup"] is not app.cleanup,
            "交给处理线程的应是标准版心和逐页设定的副本，不能和界面共用")
    d.goto(5)                                                   # 翻页、预览是只读的，照常可用
    yield d.previewed
    d.check("旋转" in d.status() or "平移" in d.status() or "无需" in d.status(), f"处理期间应能翻页预览，实际: {d.status()}")
    d.check(not d.busy() or (state(app.btn_cleanup) == "disabled" and state(app.btn_skip) == "disabled"),
            "翻页刷新之后仍应不可用")
    yield d.processed
    d.check(editable() == ["normal"] * 6, f"处理结束后各项设置应恢复，实际 {editable()}")
    d.check(state(app.btn_cleanup) == "normal" and state(app.btn_run) == "normal" and state(app.btn_cancel) == "disabled",
            "处理结束后逐页的调整、「开始处理」应恢复，「取消」不可用")
    yield d.estimated

    d.process_to("gui_lock_out2.pdf")                           # 取消之后也要恢复
    d.check(editable() == ["disabled"] * 6, "再次处理时应再次锁住")
    d.click(app.btn_cancel)
    yield lambda: not d.busy() and state(app.btn_cancel) == "disabled"
    d.check(editable() == ["normal"] * 6, f"取消之后各项设置应恢复，实际 {editable()}")

    d.open(fixtures()["v.pdf"])                                 # 分析途中取消：也要恢复，否则界面就锁死了
    d.check(editable() == ["disabled"] * 6, "打开另一本书、开始分析时应再次锁住")
    d.click(app.btn_cancel)
    yield lambda: not d.busy() and state(app.btn_cancel) == "disabled"
    d.check(editable()[:4] == ["normal"] * 4 and editable()[4:] == ["disabled"] * 2 and state(app.btn_run) == "normal",
            f"取消分析之后设置应恢复，逐页的按钮因为没有分析结果仍不可用，实际 {editable()}")
    app.var_max_angle.set(6.0)                                  # 分析参数变了：点「开始处理」会先重新分析，全程锁住
    d.process_to("gui_lock_out3.pdf")
    d.check(app.pending_run is not None and editable() == ["disabled"] * 6, "重新分析 + 处理的期间应锁住")
    yield d.processed
    d.check(editable() == ["normal"] * 6, "分析 + 处理都结束后应恢复")


def drag(d):
    """用鼠标拖版心框：四角和四边中点各有一个小方块，拖角动相邻的两条边，拖边中点只动那一条边。"""
    app = d.app
    after = app.canvases[1]

    def spot(handle):
        """小方块在「处理后」画布上的位置。handle = (水平方向的边, 竖直方向的边)，None 表示中点。"""
        return next((x, y) for h, x, y in app.handle_points(app.preview_data[2]) if h == handle)

    def pull(handle, dx, dy):
        """发真的鼠标事件：连同画布上的事件绑定一起测。"""
        x, y = (round(v) for v in spot(handle))
        after.event_generate("<ButtonPress-1>", x=x, y=y)
        after.event_generate("<B1-Motion>", x=x + dx // 2, y=y + dy // 2)
        after.event_generate("<B1-Motion>", x=x + dx, y=y + dy)
        after.event_generate("<ButtonRelease-1>", x=x + dx, y=y + dy)
        app.update()

    d.open(fixtures()["h.pdf"])
    d.check(not after.find_withtag("handle"), "分析完之前不应有可拖的小方块")
    yield d.idle
    d.goto(3)
    yield d.previewed
    d.check(len(after.find_withtag("handle")) == 8, f"红框上应有 8 个小方块，实际 {len(after.find_withtag('handle'))}")
    d.check(not app.canvases[0].find_withtag("handle"), "「处理前」一侧没有红框，也不应有小方块")
    dw, dh = app.preview_geometry[1][2:]

    pull((2, None), 30, 17)                                     # 右边的中点：只动右边，竖直方向的移动不算
    got = app.adjust.get(2)
    d.check(got is not None and abs(got[2] - 30 / dw) < 0.002 and not any((got[0], got[1], got[3])),
            f"拖右边中点应只把右边移动 {30 / dw:.4f}，实际 {got}")
    yield d.previewed
    d.check(str(app.btn_undo.cget("state")) == "normal", "拖动之后「撤销修正」应可用（提示行不再显示微调量）")
    d.check(app.store.box_adjusts(app.book["id"]).get(2) == got, "拖动的结果应保存进数据库")

    pull((0, 1), -20, -12)                                      # 左上角：左边和上边一起动，右边保持刚才的
    now = app.adjust.get(2)
    d.check(now is not None and abs(now[0] + 20 / dw) < 0.002 and abs(now[1] + 12 / dh) < 0.002
            and abs(now[2] - got[2]) < 1e-6 and now[3] == 0, f"拖左上角应移动左边和上边，实际 {now}")
    yield d.previewed

    pull((2, None), -5000, 0)                                   # 把右边拖过左边：两条边不能交叉
    yield d.previewed
    box = d.core.page_box(app.infos[2], app.ref)
    d.check(box[2] - box[0] >= 0.049, f"右边不能拖过左边，版心至少留 5% 宽，实际 {box[2] - box[0]:.3f}")
    d.click(app.btn_nudge_reset)
    yield d.previewed
    d.check(2 not in app.adjust, "复位应清掉拖动的结果")

    x, y = spot((None, 3))
    app.on_canvas_press(1, x, y - 60)                           # 点在小方块以外的地方：还是放大镜
    d.check(bool(after.find_withtag("loupe")) and app.drag is None, "点在小方块之外应显示放大镜")
    app.on_canvas_release(1)
    d.check(not after.find_withtag("loupe") and 2 not in app.adjust, "松开后放大镜消失，版心不变")

    app.var_guides.set(False)                                   # 辅助线关掉就没有红框，也就不能拖
    app.draw_preview()
    d.check(not after.find_withtag("handle"), "关掉辅助线后不应有小方块")
    app.var_guides.set(True)
    app.draw_preview()

    d.process_to("gui_drag_out.pdf")                            # 处理期间锁住：不能拖
    if d.busy():
        d.check(not after.find_withtag("handle"), "处理期间不应有可拖的小方块")
        app.on_canvas_press(1, x, y)
        d.check(app.drag is None, "处理期间不能拖版心框")
        app.on_canvas_release(1)
    yield d.processed
    yield d.previewed
    d.check(len(after.find_withtag("handle")) == 8, "处理结束后小方块应恢复")


def sr(d):
    """高清化：逐页开关，预览和输出走 AI 放大（这里用插值冒充），随书保存；程序不在时提示下载。"""
    import cv2
    core = d.core
    app = d.app
    calls = []
    def fake_sr(img, progress=None, **k):
        calls.append(1)
        if progress:
            progress(0.5)
        return cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    core.super_resolve = fake_sr
    real_find = core.find_sr_exe
    core.find_sr_exe = lambda: None
    d.open(fixtures()["mask.pdf"])
    yield d.idle
    d.goto(2)
    yield d.previewed
    d.click(app.btn_sr)
    d.check(not app.sr and any(m[1] == "高清化" for m in d.messages), "程序不在时应提示下载、不打开开关")
    core.find_sr_exe = lambda: "stub"
    d.click(app.btn_sr)
    d.check(app.sr == {1} and app.btn_sr.cget("text") == "取消高清化", "按下后这一页应标为高清化")
    yield d.previewed
    d.check(d.status().startswith("高清化完成") and calls, f"预览应经过 AI 放大、完成后有提示，实际: {d.status()}")
    d.check("高清化 1 页: 2" in app.lbl_skipped.cget("text"), f"标记列表应列出高清化的页，实际 {app.lbl_skipped.cget('text')}")
    d.check(app.store.flagged_pages(app.book["id"], "sr") == {1}, "应保存进数据库")
    out = d.process_to("gui_sr_out.pdf")
    yield d.processed
    image = fitz.open(out)[1].get_images(full=True)[0]
    d.check((image[2], image[3], image[4]) == (4 * 1240, 4 * 1754, 1), f"输出应是 4 倍分辨率的 1bit 图，实际 {image[2:5]}")
    d.click(app.btn_sr)
    d.check(app.sr == set() and app.btn_sr.cget("text") == "高清化", "再按一次取消")
    core.find_sr_exe = real_find


def align(d):
    """版心的对齐：五个按钮、手动调整（方向键、鼠标拖）、边距标注、滚动条上的黄标和「确认完毕」；只给文字书。"""
    app = d.app
    after = app.canvases[1]
    d.open(fixtures()["h.pdf"])
    yield d.idle
    d.check(5 in app.notable, f"章末半页（第 6 页）应标为值得看一眼的页，实际 {sorted(app.notable)}")
    d.goto(3)
    yield d.previewed
    d.check(all(str(b.cget("state")) == "normal" for b in app.align_buttons.values()) and str(app.btn_manual.cget("state")) == "normal",
            "分析完之后五个按钮和「手动调整」一直可用")
    d.check(str(app.btn_confirm.cget("state")) == "disabled", "整页的「确认完毕」不可用")
    d.check(len(after.find_withtag("margin")) == 8, f"应标注四条边到纸边的距离（4 条线 + 4 个数），实际 {len(after.find_withtag('margin'))}")
    std = d.core.standard_edges(app.ref)
    box = d.core.page_box(app.infos[2], app.ref)
    d.click(app.align_buttons["left"])
    d.failures.close(box[0] + app.shift[2][0], std[0], 1e-6, "「靠左」后红框左边应贴齐标准版心的左边")
    d.click(app.align_buttons["bottom"])
    d.failures.close(box[3] + app.shift[2][1], std[3], 1e-6, "「靠下」后红框下边应贴齐标准版心的下边")
    d.failures.close(box[0] + app.shift[2][0], std[0], 1e-6, "「靠下」不应动水平方向")
    d.click(app.align_buttons["center"])
    d.failures.close((box[0] + box[2]) / 2 + app.shift[2][0], 0.5, 1e-6, "「居中」后水平居中")
    d.failures.close((box[1] + box[3]) / 2 + app.shift[2][1], 0.5, 1e-6, "「居中」后竖直居中")
    yield d.previewed

    d.click(app.btn_manual)                                     # 手动调整：方向键
    d.check(app.manual_move and app.btn_manual.cget("text") == "结束手动调整", "按下后进入手动调整")
    before = app.shift[2]
    d.press(after, "<Right>")
    d.press(after, "<Down>")
    d.check(abs(app.shift[2][0] - before[0] - d.gui.NUDGE_STEP) < 1e-9 and abs(app.shift[2][1] - before[1] - d.gui.NUDGE_STEP) < 1e-9,
            f"方向键应移动版心一步，实际 {before} → {app.shift[2]}")
    yield d.previewed
    x0, y0, dw, dh = app.preview_geometry[1]                    # 鼠标拖：在页面上按住拖 20 像素
    x, y = x0 + dw // 2, y0 + dh // 2
    before = app.shift[2]
    after.event_generate("<ButtonPress-1>", x=x, y=y)
    after.event_generate("<B1-Motion>", x=x + 10, y=y + 5)
    after.event_generate("<B1-Motion>", x=x + 20, y=y + 5)
    after.event_generate("<ButtonRelease-1>", x=x + 20, y=y + 5)
    app.update()
    d.check(abs(app.shift[2][0] - before[0] - 20 / dw) < 0.002 and abs(app.shift[2][1] - before[1] - 5 / dh) < 0.002,
            f"拖动应按像素移动版心，实际 {before} → {app.shift[2]}")
    d.click(app.btn_manual)
    d.check(not app.manual_move, "再按一次结束手动调整")
    d.check(app.store.shifts(app.book["id"]).get(2) == app.shift[2], "平移量应保存进数据库")
    d.click(app.btn_nudge_reset)
    d.check(2 not in app.shift, "复位应清掉指定的平移量")

    d.goto(6)                                                   # 黄标的页：确认完毕
    yield d.previewed
    d.check(str(app.btn_confirm.cget("state")) == "normal", "翻到黄标的页「确认完毕」可用")
    marks = len(app.scroll.find_all())
    d.click(app.align_buttons["top"])
    d.click(app.btn_confirm)
    d.check(5 in app.confirmed and str(app.btn_confirm.cget("state")) == "disabled" and len(app.scroll.find_all()) == marks - 1,
            "确认后按钮变灰、滚动条上少一个黄标")
    d.check(app.lbl_nudge.cget("text") == "", "提示行不显示「版心位置已指定」之类的文字")
    d.check(app.store.flagged_pages(app.book["id"], "confirmed") == {5}, "确认应保存进数据库")
    d.click(app.btn_undo)
    d.check(5 not in app.confirmed and 5 not in app.shift and str(app.btn_confirm.cget("state")) == "normal"
            and len(app.scroll.find_all()) == marks, "「撤销修正」应清掉指定的位置、退回未确认、黄标回来")
    app.scroll_to(app.scroll.winfo_height() // 2)               # 拖滚动条翻页
    d.check(app.var_page.get() in (4, 5), f"滚动条中点应翻到第 4～5 页，实际 {app.var_page.get()}")

    app.var_book.set("manga")                                   # 漫画：这些都不显示
    app.update()
    d.check(not app.align_group.winfo_manager(), "漫画书不显示版心对齐那一组")
    yield d.previewed
    d.check(not after.find_withtag("margin"), "漫画书不标注边距")
    app.var_book.set("text")


def keystone(d):
    """梯形校正：按钮在「梯形校正 / 执行校正」之间切换；四个角在左侧可拖；执行后铺满整页；取消校正恢复原样；随书保存。"""
    app = d.app
    before = app.canvases[0]
    d.open(fixtures()["trap.pdf"])
    d.check(str(app.btn_keystone.cget("state")) == "disabled", "分析完之前「梯形校正」不可用")
    yield d.idle
    d.goto(2)
    yield d.previewed
    d.check(str(app.btn_keystone.cget("state")) == "normal" and not app.btn_keystone_cancel.winfo_manager(),
            "分析完之后「梯形校正」可用，没校正过的页没有「取消校正」")
    old_status = d.status()
    d.click(app.btn_keystone)
    d.check(app.btn_keystone.cget("text") == "执行校正" and app.keystone_edit is not None and app.keystone_edit[0] == 1,
            "按下后应进入编辑、按钮改名为「执行校正」")
    d.check(bool(app.btn_keystone_cancel.winfo_manager()), "编辑中应有「取消校正」")
    yield d.previewed
    d.check(len(before.find_withtag("quad_handle")) == 4, f"左侧应画出四边形和 4 个角，实际 {len(before.find_withtag('quad_handle'))}")
    d.check("梯形校正" in d.status(), f"右侧应显示校正后的结果，实际: {d.status()}")
    quad0 = [list(p) for p in app.keystone_edit[1]]
    x0, y0, dw, dh = app.preview_geometry[0]
    hx, hy = (round(v) for v in app.quad_points()[0])          # 拖左上角
    before.event_generate("<ButtonPress-1>", x=hx, y=hy)
    before.event_generate("<B1-Motion>", x=hx - 10, y=hy - 8)
    before.event_generate("<ButtonRelease-1>", x=hx - 10, y=hy - 8)
    app.update()
    moved = app.keystone_edit[1][0]
    d.check(abs(moved[0] - (quad0[0][0] - 10 / dw)) < 0.002 and abs(moved[1] - (quad0[0][1] - 8 / dh)) < 0.002
            and app.keystone_edit[1][1:] == quad0[1:], f"拖左上角应只动左上角，实际 {app.keystone_edit[1]}")
    d.click(app.btn_keystone_cancel)                            # 编辑中的「取消校正」：放弃编辑
    d.check(app.keystone_edit is None and app.btn_keystone.cget("text") == "梯形校正" and 1 not in app.keystone,
            "编辑中点「取消校正」应放弃编辑")
    d.click(app.btn_keystone)                                   # 重新进入编辑（自动的四边形），直接执行
    d.click(app.btn_keystone)
    d.check(app.btn_keystone.cget("text") == "梯形校正" and app.keystone_edit is None and 1 in app.keystone,
            "执行后按钮应改回「梯形校正」，校正记在这一页上")
    d.check(bool(app.btn_keystone_cancel.winfo_manager()), "校正过的页应有「取消校正」")
    d.check(app.store.keystones(app.book["id"]).get(1) == app.keystone[1], "梯形校正应保存进数据库")
    d.check("梯形校正 1 页: 2" in app.lbl_skipped.cget("text"), f"标记列表应列出梯形校正的页，实际 {app.lbl_skipped.cget('text')}")
    yield d.previewed
    d.check(not before.find_withtag("quad") and app.preview_data[2] is None, "执行后左侧的四边形消失，右侧没有红框")
    d.goto(1)
    yield d.previewed
    d.check(not app.btn_keystone_cancel.winfo_manager() and app.btn_keystone.cget("text") == "梯形校正", "别的页没有「取消校正」")
    d.click(app.btn_keystone)                                   # 进入编辑再翻页：放弃编辑
    d.goto(2)
    yield d.previewed
    d.check(app.keystone_edit is None and 0 not in app.keystone, "翻页应放弃没执行的编辑")

    out = d.process_to("gui_keystone_out.pdf")
    yield d.processed
    m = measure(out)[1]
    d.check(m["left"] < 0.05 and m["right"] < 0.05 and abs(m["angle"]) < 0.3, f"输出的第 2 页应铺满整页、行大致是平的（自动的四个角不如手调的准），实际 {m}")
    d.check("梯形校正" in d.log(), "处理日志里应写明梯形校正")

    d.open(fixtures()["trap.pdf"])                              # 重新打开：恢复
    d.goto(2)
    yield d.previewed
    d.check(1 in app.keystone and bool(app.btn_keystone_cancel.winfo_manager()), "重新打开应恢复梯形校正")
    d.click(app.btn_keystone_cancel)
    d.check(1 not in app.keystone and not app.btn_keystone_cancel.winfo_manager(), "「取消校正」应清掉这一页的校正")
    yield d.previewed
    d.check(d.status() == old_status, f"取消后应恢复原样，实际: {d.status()}")
    d.check(app.store.keystones(app.book["id"]) == {}, "取消应写进数据库")


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


def book(d):
    """书的类型：文字书/漫画书显示不同的选项、给不同的建议；类型和「纸面找平」随书保存；换类型不用重新分析。"""
    app = d.app
    shown = lambda: [n for n, c in app.option_checks.items() if c.winfo_manager()]
    d.open(fixtures()["manga.pdf"], analyze=False)
    d.check(app.var_book.get() == "text" and "enhance" in shown() and "flatten" not in shown(),
            f"默认是文字书，应显示「显示增强」、不显示「纸面找平」，实际 {shown()}")
    app.var_book.set("manga")
    d.check(app.var_book_label.get() == "漫画书", "下拉框的文字应跟着类型变")
    d.check("flatten" in shown() and "enhance" not in shown() and "upscale" not in shown(),
            f"漫画书应显示「纸面找平」、不显示黑白页的两项，实际 {shown()}")
    d.check(not d.busy(), "换类型不需要重新分析")
    d.click(app.btn_reanalyze)
    yield d.idle
    yield d.previewed
    d.check("纸面找平: 建议 开启" in app.rec_labels["flatten"].cget("text") and app.rec_labels["flatten"].winfo_manager(),
            f"漫画有灰影的页多，应建议开启「纸面找平」，实际: {app.rec_labels['flatten'].cget('text')}")
    d.check("「纸面找平」建议开启" in d.log() and "「显示增强」" not in d.log(), "漫画的日志里应有找平的说明、没有显示增强的说明")
    d.check(not app.var_flatten.get() and app.btn_recommend.cget("text") == "采用建议值", "开关类的建议也走同一个按钮")
    d.click(app.btn_recommend)
    d.check(app.var_flatten.get(), "「采用建议值」应把「纸面找平」勾上")
    yield d.previewed
    d.check("纸面找平" in d.status(), f"有灰影的页处理时应做找平，实际: {d.status()}")
    d.goto(5)
    yield d.previewed
    d.check("纸面找平（灰影 0 级）" in d.status(), f"纸面均匀的页也找平（页与页才一致），实际: {d.status()}")
    d.goto(1)
    app.var_enhance.set(True)                                   # 藏起来的选项就算勾着也不生效
    d.check(not d.app.get_options().enhance and d.app.get_options().flatten, "漫画书下「显示增强」应视为关闭")

    app.var_book.set("text")
    d.check(not d.busy() and app.infos is not None, "换回文字书也不用重新分析")
    yield d.previewed
    d.check(not app.rec_labels["flatten"].winfo_manager() and "纸面找平" not in d.status(),
            "文字书不显示找平的建议，也不做找平")
    app.var_book.set("manga")
    out = d.process_to("gui_book_out.pdf")
    yield d.processed
    d.check("纸面找平" in d.log(), "处理日志里应写明找平")
    edge = page_array(out, 0, dpi=72)                           # 第 1 页右侧的灰影处理后应变白
    d.check(np.percentile(edge[:, -6:], 90) >= 248, f"找平后页边应接近纯白，实际 {np.percentile(edge[:, -6:], 90)}")

    d.open(fixtures()["h.pdf"])
    yield d.idle
    d.check(app.var_book.get() == "text" and not app.var_flatten.get(), "换一本书应回到默认的类型")
    d.open(fixtures()["manga.pdf"])
    d.check(app.var_book.get() == "manga" and app.var_flatten.get() and not d.busy(),
            f"重新打开漫画应恢复它的类型和选项，实际 {app.var_book.get()} {app.var_flatten.get()}")


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
    ("enhance", enhance, "gui_enhance.db", True, None),
    ("lock", lock, "gui_lock.db", True, None),
    ("drag", drag, "gui_drag.db", True, None),
    ("keystone", keystone, "gui_keystone.db", True, None),
    ("align", align, "gui_align.db", True, None),
    ("sr", sr, "gui_sr.db", True, None),
    ("files", files, "gui_files.db", True, None),
    ("book", book, "gui_book.db", True, None),
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
