# -*- coding: utf-8 -*-
"""自动驱动图形界面的工具。

场景写成生成器：顺序往下写，需要等的时候 yield 一个「条件」（无参函数，返回真值表示等到了）
或者一个毫秒数。Driver 在 tk 的事件循环里轮询这个条件，满足了再让场景继续往下走：

    def scenario(d):
        d.open(path)
        yield d.idle                 # 等全书分析完
        d.goto(4)
        yield d.previewed            # 等预览刷新
        d.check("旋转" in d.status(), "……")

每个场景都有总超时：界面测试一旦出错很容易卡在事件循环里不退出（吃过亏）。
"""
import os
import time
import traceback
from tkinter import messagebox

from harness import Failures, work_path

POLL_MS = 120


class Driver:
    def __init__(self, db_name, timeout=300):
        os.environ["PDF_RESHAPE_DB"] = work_path(db_name)      # 绝不碰用户真实的进度数据库
        import pdf_reshape_gui as gui
        self.gui, self.core = gui, gui.core
        self.failures = Failures()
        self.messages = []                                      # 被拦下来的弹窗 (种类, 标题, 内容)
        messagebox.showinfo = lambda t, m, **k: self.messages.append(("info", t, m))
        messagebox.showwarning = lambda t, m, **k: self.messages.append(("warning", t, m))
        messagebox.showerror = lambda t, m, **k: self.messages.append(("error", t, m))
        messagebox.askyesno = lambda t, m, **k: True
        self.app = gui.App()
        self.app.geometry("1250x860")
        self.app.lift()
        self.deadline = time.time() + timeout
        self.finished = False                                   # 收到过「处理完成」
        self.preview_seen = 0                                   # 最近一次收到的预览属于第几次请求
        self.estimate_seen = 0
        handle = self.app._handle

        def spy(msg):
            if msg[0] == "done":
                self.finished = True
            elif msg[0] == "preview":
                self.preview_seen = msg[1]
            elif msg[0] == "estimate":
                self.estimate_seen = msg[1]
            handle(msg)
        self.app._handle = spy

    # ------------------------------------------------------------ 运行

    def run(self, scenario):
        steps = scenario(self)
        waiting = [None]

        def tick():
            try:
                if time.time() > self.deadline:
                    raise TimeoutError("场景超时")
                cond = waiting[0]
                if cond is not None and not (time.time() >= cond if isinstance(cond, float) else cond()):
                    self.app.after(POLL_MS, tick)
                    return
                step = next(steps)
                waiting[0] = time.time() + step / 1000 if isinstance(step, (int, float)) else step
                self.app.after(POLL_MS, tick)
            except StopIteration:
                self.close()
            except Exception:
                self.failures.append(traceback.format_exc().rstrip())
                self.close()

        self.app.after(300, tick)
        self.app.mainloop()
        return self.failures

    def close(self):
        try:
            self.app.on_close()
        except Exception:
            self.app.destroy()

    # ------------------------------------------------------------ 等待的条件

    def busy(self):
        return bool(self.app.worker and self.app.worker.is_alive())

    def idle(self):
        """全书分析（或处理）已经结束。"""
        return self.app.infos is not None and not self.busy()

    def previewed(self):
        """最近一次预览请求已经显示出来。"""
        return (self.app.preview_after_id is None and self.app.preview_data is not None
                and self.preview_seen == self.app.preview_gen)

    def estimated(self):
        """输出大小的预估已经算完。"""
        return (self.app.estimate_after_id is None and self.estimate_seen == self.app.estimate_gen
                and self.app.lbl_estimate.cget("text").startswith("预计输出约"))

    def processed(self):
        return self.finished and not self.busy()

    # ------------------------------------------------------------ 操作与读取

    def check(self, condition, message):
        return self.failures.check(condition, message)

    def open(self, path, analyze=True):
        """打开文件。打开后程序不会自动分析（用户要求），analyze=True 时替用户点一下「分析全书」——
        有保存的分析结果的书直接恢复，不用点。"""
        self.app.open_file(path)
        self.app.update()
        if analyze and self.app.infos is None and self.app.page_count:
            self.click(self.app.btn_reanalyze)

    def goto(self, page):
        self.app.var_page.set(page)
        self.app.request_preview()
        self.app.update()

    def refresh(self):
        self.app.request_preview()
        self.app.update()

    def click(self, button):
        button.invoke()
        self.app.update()

    def process_to(self, name):
        """点「开始处理」，输出到工作目录下的 name。"""
        out = work_path(name)
        self.finished = False
        self.app.var_output.set(out)
        self.app.run()
        return out

    def status(self):
        return self.app.lbl_page_status.cget("text")

    def log(self):
        return self.app.log.get("1.0", "end").strip()

    def options(self):
        names = ("book", "deskew", "center", "per_page", "clean", "upscale", "enhance", "flatten",
                 "max_angle", "min_angle", "quality", "dpi")
        return {n: getattr(self.app, "var_" + n).get() for n in names}

    def frame_after(self, index):
        """第 index 页（从 0 开始）处理后红框的位置。"""
        info = self.app.infos[index]
        box = self.core.page_box(info, self.app.ref)
        sx, sy = self.core.compute_shift(info, self.app.ref, False)
        return [round(box[0] + sx, 3), round(box[1] + sy, 3), round(box[2] + sx, 3), round(box[3] + sy, 3)]

    def find_widget(self, widget_class, variable, text=None):
        """按控件类型和绑定的变量找控件（比如「小于此角度不旋转」的数值框）；没有绑定变量的按文字开头找。"""
        stack = [self.app]
        while stack:
            w = stack.pop()
            stack.extend(w.winfo_children())
            if w.winfo_class() != widget_class:
                continue
            if str(w.cget("text")).startswith(text) if text else str(w.cget("textvariable")) == str(variable):
                return w
        raise LookupError(f"找不到绑定 {variable or text} 的 {widget_class}")

    def press(self, widget, key):
        widget.focus_force()
        self.app.update()
        widget.event_generate(key)
        self.app.update()
