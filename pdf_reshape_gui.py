#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""扫描书籍 PDF 整形工具（图形界面）。核心处理在 pdf_reshape.py。

启动:  python pdf_reshape_gui.py [input.pdf]
"""
import base64
import queue
import sys
import threading
import time
from datetime import datetime
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import fitz
import numpy as np

import pdf_reshape as core
from pdf_reshape_store import Store

PREVIEW_MAX_SIDE = 1600   # 预览用图像的长边上限（像素）
LOUPE_SIZE = 300          # 放大镜的边长（屏幕像素）
LOUPE_SPAN = 0.07         # 放大镜里显示页面宽度的 7%
HANDLE_HALF = 4           # 版心框上拖动用的白色小方块：边长的一半（屏幕像素）
HANDLE_REACH = 9          # 鼠标离小方块的中心不到这么远就算点中（比方块本身大一圈，好点）
MIN_BOX = 0.05            # 拖动时版心至少留这么宽（高），和 core.page_box 的限制一致
# 八个小方块各管哪几条边：(水平方向的边, 竖直方向的边)，数字是版心 (左, 上, 右, 下) 里的下标，
# None = 这个方向不动（边的中点）。四个角动相邻的两条边，四个中点只平移自己那条边
HANDLES = ((0, 1), (None, 1), (2, 1), (0, None), (2, None), (0, 3), (None, 3), (2, 3))
HANDLE_CURSORS = {(0, 1): "size_nw_se", (2, 3): "size_nw_se", (2, 1): "size_ne_sw", (0, 3): "size_ne_sw",
                  (0, None): "size_we", (2, None): "size_we", (None, 1): "size_ns", (None, 3): "size_ns"}
NUDGE_STEP = 0.002        # 版心边框每按一次移动页面宽（高）的 0.2%
ESTIMATE_DELAY = 800      # 设置变了之后等这么久（毫秒）再重新预估输出大小
NUDGE_ESTIMATE_DELAY = 2000   # 调版心边框时等得更久：停手 2 秒之后才重新预估
EDGES = {"上": 1, "下": 3, "左": 0, "右": 2}       # 边 → 版心 (左, 上, 右, 下) 里的下标


class App(tk.Tk):
    def __init__(self, initial_file=None):
        super().__init__()
        self.title("PDF 整形 — 纠偏 / 版心居中")
        self.geometry("1180x820")
        self.minsize(900, 600)

        self.msgs = queue.Queue()          # 工作线程 → 界面
        self.cancel_event = threading.Event()
        self.worker = None                 # 分析/处理线程（同时只有一个）
        self.page_count = 0
        self.infos = None                  # 全书分析结果
        self.ref = None                    # 标准版心统计
        self.analysis_key = None           # 产生 infos 时的 (文件, 检测范围, DPI)
        self.analyzing_key = None
        self.pending_run = None            # 分析结束后要接着执行的处理任务
        self.preview_gen = 0               # 预览请求序号，用来丢弃过期结果
        self.preview_data = None           # (before, after, box, 本页是否已删除)
        self.preview_index = None          # preview_data 是第几页的（从 0 开始）
        self.preview_shown_gen = 0         # preview_data 属于第几次预览请求
        self.drag = None                   # 正在拖版心框：(小方块, 按下时的 x, y, 按下时的红框)
        self.drag_box = None               # 拖到现在红框的样子
        self.preview_after_id = None
        self.preview_full = None           # (处理前, 处理后, 平移量)：原尺寸的图，放大镜用
        self.preview_geometry = [None, None]   # 两个预览画布上页面图像的位置 (x0, y0, 宽, 高)
        self.photos = [None, None]
        self.loupe_photos = [None, None]
        self.book = None                   # 数据库里这本书的记录
        self.skipped = set()               # 用户指定「本页不修正」的页（从 0 开始）
        self.deleted = set()               # 用户指定「删除当前页」的页：新 PDF 中不输出
        self.adjust = {}                   # 用户对版心边框的手动微调 {页序: (左, 上, 右, 下 的移动量)}
        self.cleanup = set()               # 用户指定「去除边缘污染」的页
        self.align = {}                    # 用户点「版心居中」那一刻的 adjust：对齐按它算（见 core.align_box）
        self.estimate_gen = 0              # 输出大小预估的请求序号（丢弃过期结果用）
        self.estimate_key = None           # 上次预估时的全部相关设置；没变就不重算
        self.estimate_cache = {}           # 抽样页的编码结果，多次预估之间复用
        self.estimate_after_id = None
        self.nudged_at = 0.0               # 最近一次调版心边框的时刻（time.monotonic）
        self.estimate_running = False      # 有一次预估正在算
        self.locked = False                # 分析、处理期间：会改变输出的控件都不可用
        self.lockable = []                 # 平时一直可用、锁住时才不可用的控件（其余的各有各的条件）
        try:
            self.store = Store()
        except Exception as e:             # 数据库打不开也不影响整形本身，只是不能保存进度
            self.store = None
            self.after(200, lambda: messagebox.showwarning("进度无法保存", f"打不开进度数据库：{e}"))
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self._build_ui()
        self.after(50, self._poll)
        if self.store:                      # 启动时清理历史记录：原文件已经不在了的书，记录一并删掉
            try:
                for path in self.store.prune_missing():
                    self.write_log(f"原文件已不存在，删除了它的历史记录：{path}")
            except Exception as e:
                self.write_log(f"清理历史记录时出错（不影响使用）：{e}")
        if initial_file:
            self.open_file(initial_file)

    # ------------------------------------------------------------ 界面

    def _build_ui(self):
        pad = dict(padx=6, pady=3)

        files = ttk.Frame(self)
        files.pack(fill="x", padx=8, pady=(8, 0))
        files.columnconfigure(1, weight=1)
        self.var_input = tk.StringVar()
        self.var_output = tk.StringVar()
        ttk.Label(files, text="输入 PDF").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(files, textvariable=self.var_input, state="readonly").grid(row=0, column=1, sticky="ew", **pad)
        lock = self.lockable.append
        lock(ttk.Button(files, text="打开…", command=self.browse_input))
        self.lockable[-1].grid(row=0, column=2, sticky="ew", **pad)
        lock(ttk.Button(files, text="历史记录…", command=self.show_history))
        self.lockable[-1].grid(row=0, column=3, **pad)
        ttk.Label(files, text="输出 PDF").grid(row=1, column=0, sticky="w", **pad)
        lock(ttk.Entry(files, textvariable=self.var_output))
        self.lockable[-1].grid(row=1, column=1, sticky="ew", **pad)
        lock(ttk.Button(files, text="另存为…", command=self.browse_output))
        self.lockable[-1].grid(row=1, column=2, sticky="ew", **pad)

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, padx=8, pady=4)

        # ---- 左侧：选项
        side = ttk.Frame(body)
        side.pack(side="left", fill="y", padx=(0, 8))

        # 书的类型：决定「修正内容」里显示哪些选项、分析完给哪些建议。存的是 core.BOOK_TYPES 的键
        kind = ttk.Frame(side)
        kind.pack(fill="x", pady=(0, 6))
        ttk.Label(kind, text="书的类型").pack(side="left", padx=(6, 8))
        self.var_book = tk.StringVar(value="text")
        self.var_book_label = tk.StringVar(value=core.BOOK_TYPES["text"])
        self.book_box = ttk.Combobox(kind, textvariable=self.var_book_label, values=list(core.BOOK_TYPES.values()),
                                     width=8, state="readonly")
        self.book_box.pack(side="left")
        lock(self.book_box)
        self.book_box.bind("<<ComboboxSelected>>", lambda e: self.var_book.set(
            next(k for k, v in core.BOOK_TYPES.items() if v == self.var_book_label.get())))
        self.var_book.trace_add("write", lambda *_: self.apply_book_type())

        box = ttk.LabelFrame(side, text="修正内容")
        box.pack(fill="x", pady=(0, 6))
        self.var_deskew = tk.BooleanVar(value=True)
        self.var_center = tk.BooleanVar(value=True)
        self.var_per_page = tk.BooleanVar(value=False)
        self.var_clean = tk.BooleanVar(value=False)
        self.var_upscale = tk.BooleanVar(value=False)
        self.var_enhance = tk.BooleanVar(value=False)
        self.var_flatten = tk.BooleanVar(value=False)
        self.option_checks = {}             # {Options 字段名: 勾选框}，按书的类型决定显示哪些（apply_book_type）
        for name, text, var in [("deskew", "倾斜校正", self.var_deskew),
                                ("center", "版心居中", self.var_center),
                                ("per_page", "每页各自居中\n（不参照全书标准版心）", self.var_per_page),
                                ("clean_margin", "版心外涂成纸色\n（去黑边、阴影）", self.var_clean),
                                ("upscale", "黑白页旋转时 2 倍分辨率\n（笔画更平滑，体积变大）", self.var_upscale),
                                ("enhance", "显示增强\n（只增强能明显改善的页）", self.var_enhance),
                                ("flatten", "纸面找平\n（纸色拉白，去掉页边的灰影）", self.var_flatten)]:
            self.option_checks[name] = ttk.Checkbutton(box, text=text, variable=var, command=self.schedule_preview)
            lock(self.option_checks[name])
        self.option_pad = pad
        self.apply_book_type()

        box = ttk.LabelFrame(side, text="参数")
        box.pack(fill="x", pady=(0, 6))
        self.var_max_angle = tk.DoubleVar(value=5.0)
        self.var_min_angle = tk.DoubleVar(value=0.1)
        self.var_quality = tk.IntVar(value=90)
        self.var_dpi = tk.IntVar(value=300)
        rows = [("倾斜检测范围 ±°", self.var_max_angle, 1, 15, 1),
                ("小于此角度不旋转 °", self.var_min_angle, 0, 2, 0.05),
                ("JPEG 质量", self.var_quality, 50, 100, 5),
                ("渲染 DPI", self.var_dpi, 100, 600, 50)]
        for r, (text, var, lo, hi, step) in enumerate(rows):
            ttk.Label(box, text=text).grid(row=r, column=0, sticky="w", **pad)
            lock(ttk.Spinbox(box, textvariable=var, from_=lo, to=hi, increment=step, width=6,
                             command=self.schedule_preview))
            self.lockable[-1].grid(row=r, column=1, **pad)

        # 分析完全书后在这里显示各个参数的建议值（每项一行提示，没有建议的不显示）
        self.rec = {}                       # {"min_angle" / "quality" / "max_angle" / "dpi": 建议值或 None}
        self.rec_labels = {}
        for k, name in enumerate(("max_angle", "min_angle", "quality", "dpi", "flatten")):
            label = ttk.Label(box, text="", foreground="#0a58ca", wraplength=190, justify="left")
            self.rec_labels[name] = label
        self.rec_button_row = len(rows) + 5
        # 一个按钮管所有的建议值：「采用建议值」↔「恢复默认值」
        self.btn_recommend = ttk.Button(box, text="采用建议值", command=self.toggle_recommendation)
        lock(self.btn_recommend)

        box = ttk.LabelFrame(side, text="处理范围")
        box.pack(fill="x", pady=(0, 6))
        self.var_pages = tk.StringVar()
        lock(ttk.Entry(box, textvariable=self.var_pages, width=22))
        self.lockable[-1].pack(fill="x", **pad)
        ttk.Label(box, text="例: 1-20 或 3,5,8-12\n留空 = 全部页", foreground="#666").pack(anchor="w", **pad)

        # 打开文件后不自动分析（用户要求：先选类型、设好参数）。没分析过时叫「分析全书」，分析过了叫「重新分析」
        self.btn_reanalyze = ttk.Button(side, text="分析全书", command=self.reanalyze)
        self.btn_run = ttk.Button(side, text="开始处理", command=self.run, state="disabled")
        self.btn_run.pack(fill="x", pady=(6, 3), ipady=6)
        self.btn_cancel = ttk.Button(side, text="取消", command=self.cancel, state="disabled")
        self.btn_cancel.pack(fill="x")
        self.lbl_estimate = ttk.Label(side, text="", foreground="#0a58ca", wraplength=200, justify="left")
        self.lbl_estimate.pack(fill="x", pady=(8, 0))

        # ---- 右侧：预览
        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True)

        nav = ttk.Frame(right)
        nav.pack(fill="x")
        ttk.Button(nav, text="◀", width=3, command=lambda: self.step_page(-1)).pack(side="left")
        self.var_page = tk.IntVar(value=1)
        spin = ttk.Spinbox(nav, textvariable=self.var_page, from_=1, to=1, width=6,
                           command=self.schedule_preview)
        spin.pack(side="left", padx=4)
        spin.bind("<Return>", lambda e: self.schedule_preview())
        self.spin_page = spin
        self.lbl_total = ttk.Label(nav, text="/ 0")
        self.lbl_total.pack(side="left")
        ttk.Button(nav, text="▶", width=3, command=lambda: self.step_page(1)).pack(side="left", padx=4)
        self.var_guides = tk.BooleanVar(value=True)
        ttk.Checkbutton(nav, text="辅助线", variable=self.var_guides,
                        command=self.draw_preview).pack(side="left", padx=12)
        # 按下＝当前页不做修正、原样保留；再按一次恢复。只影响当前页
        self.var_skip = tk.BooleanVar(value=False)      # 当前页是不是「不修正」（按钮的文字跟着它变）
        self.btn_skip = ttk.Button(nav, text="本页不修正", command=self.toggle_skip, state="disabled")
        self.btn_skip.pack(side="left", padx=4)
        # 删除当前页：新 PDF 中不输出这一页（原文件不动）。按下后变成「恢复当前页」
        style = ttk.Style(self)
        style.configure("Delete.TButton", foreground="#d00000")
        style.configure("Restore.TButton", foreground="#1a7f37")
        self.btn_delete = ttk.Button(nav, text="删除当前页", style="Delete.TButton",
                                     command=self.toggle_delete, state="disabled")
        self.btn_delete.pack(side="left", padx=4)
        self.lbl_skipped = ttk.Label(nav, text="", foreground="#666")
        self.lbl_skipped.pack(side="left", padx=4)
        self.lbl_page_status = ttk.Label(nav, text="", foreground="#0a58ca")
        self.lbl_page_status.pack(side="left", padx=8)

        # ---- 第二行：手动微调当前页的版心边框（预览里的红框）
        edge_bar = ttk.Frame(right)
        edge_bar.pack(fill="x", pady=(4, 0))
        ttk.Label(edge_bar, text="版心边框").pack(side="left")
        self.var_edge = tk.StringVar(value="上")
        edge_box = ttk.Combobox(edge_bar, textvariable=self.var_edge, values=list(EDGES), width=4, state="readonly")
        edge_box.pack(side="left", padx=(6, 10))
        edge_box.bind("<<ComboboxSelected>>", lambda e: self.update_nudge_buttons())
        self.nudge_buttons = {}
        # 按钮上画三角形：↑↓ 这两个字符会被 Windows 固定换成彩色的表情符号，和 ←→ 不一致
        glyphs = {"↑": "▲", "↓": "▼", "←": "◀", "→": "▶"}
        for arrow, delta in (("↑", -1), ("↓", 1), ("←", -1), ("→", 1)):
            # tk.Button 而不是 ttk：按住不放可以连续调（repeatdelay / repeatinterval）
            btn = tk.Button(edge_bar, text=glyphs[arrow], width=3, relief="groove", state="disabled",
                            repeatdelay=400, repeatinterval=80, command=lambda d=delta: self.nudge(d))
            btn.pack(side="left", padx=1)
            self.nudge_buttons[arrow] = btn
        self.btn_nudge_reset = ttk.Button(edge_bar, text="复位", width=5, command=self.reset_nudge, state="disabled")
        self.btn_nudge_reset.pack(side="left", padx=(10, 2))
        # 调红框本身不移动页面；点这个按钮才按现在的红框重新对齐。「复位」把两者一起撤销
        self.btn_center = ttk.Button(edge_bar, text="版心居中", width=9, command=self.center_by_box, state="disabled")
        self.btn_center.pack(side="left", padx=(2, 6))
        # 红框照邻页的来：当前页的红框判断得不好、邻页的好时用。结果记成手动微调，可以再调、可以复位
        self.btn_like_prev = ttk.Button(edge_bar, text="与前页相同", state="disabled",
                                        command=lambda: self.copy_neighbor_box(-1))
        self.btn_like_prev.pack(side="left", padx=(6, 2))
        self.btn_like_next = ttk.Button(edge_bar, text="与后页相同", state="disabled",
                                        command=lambda: self.copy_neighbor_box(1))
        self.btn_like_next.pack(side="left", padx=2)
        # 去除边缘污染：把红框外疑似墨迹的地方按周围干净的纸面重新画上。逐页的开关
        self.btn_cleanup = ttk.Button(edge_bar, text="去除边缘污染", state="disabled", command=self.toggle_cleanup)
        self.btn_cleanup.pack(side="left", padx=(12, 6))
        self.lbl_nudge = ttk.Label(edge_bar, text="", foreground="#666")
        self.lbl_nudge.pack(side="left", padx=4)

        panes = ttk.Frame(right)
        panes.pack(fill="both", expand=True, pady=4)
        panes.columnconfigure((0, 1), weight=1, uniform="p")
        panes.rowconfigure(1, weight=1)
        self.canvases = []
        for c, title in enumerate(("处理前", "处理后")):
            ttk.Label(panes, text=title).grid(row=0, column=c)
            cv = tk.Canvas(panes, background="#808080", highlightthickness=0)
            cv.grid(row=1, column=c, sticky="nsew", padx=2)
            cv.bind("<Configure>", lambda e: self.draw_preview())
            # 点一下预览图，方向键就回来翻页；按住不放是放大镜（两边同时放大同一处，对比增强前后）
            # 「处理后」一侧按在红框的小方块上则是拖版心框
            cv.bind("<ButtonPress-1>", lambda e, c=c: (e.widget.focus_set(), self.on_canvas_press(c, e.x, e.y)))
            cv.bind("<B1-Motion>", lambda e, c=c: self.on_canvas_motion(c, e.x, e.y))
            cv.bind("<ButtonRelease-1>", lambda e, c=c: self.on_canvas_release(c))
            if c == 1:
                cv.bind("<Motion>", lambda e: self.update_cursor(e.x, e.y))
            self.canvases.append(cv)
        # 键盘翻页：PageUp/PageDown、上下左右方向键、Home/End
        for key, delta in (("<Prior>", -1), ("<Next>", 1), ("<Up>", -1), ("<Down>", 1),
                           ("<Left>", -1), ("<Right>", 1), ("<Home>", -10**9), ("<End>", 10**9)):
            self.bind(key, lambda e, d=delta: self.on_page_key(e, d))

        # ---- 底部：进度和日志
        bottom = ttk.Frame(self)
        bottom.pack(fill="x", padx=8, pady=(0, 8))
        self.progress = ttk.Progressbar(bottom, maximum=100)
        self.progress.pack(fill="x")
        self.lbl_status = ttk.Label(bottom, text="请打开一个扫描 PDF")
        self.lbl_status.pack(anchor="w", pady=2)
        log_frame = ttk.Frame(bottom)
        log_frame.pack(fill="x")
        self.log = tk.Text(log_frame, height=7, state="disabled", wrap="none", font="TkDefaultFont")
        scroll = ttk.Scrollbar(log_frame, command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        self.log.pack(side="left", fill="x", expand=True)
        scroll.pack(side="left", fill="y")

    # ------------------------------------------------------------ 选项读取

    def get_options(self):
        def num(var, default, cast):
            try:
                return cast(var.get())
            except (tk.TclError, ValueError):
                return default
        return core.Options(
            book=self.var_book.get(),
            deskew=self.var_deskew.get(), center=self.var_center.get(),
            per_page=self.var_per_page.get(), clean_margin=self.var_clean.get(),
            upscale=self.var_upscale.get(), enhance=self.var_enhance.get(), flatten=self.var_flatten.get(),
            max_angle=max(num(self.var_max_angle, 5.0, float), 0.5),
            min_angle=max(num(self.var_min_angle, 0.1, float), 0.0),
            dpi=min(max(num(self.var_dpi, 300, int), 72), 1200),
            quality=min(max(num(self.var_quality, 90, int), 1), 100)).effective()   # 藏起来的选项一律按关闭

    def apply_book_type(self):
        """按书的类型显示对应的「修正内容」选项；建议值的提示也跟着换。藏起来的勾选框可能还勾着，
        Options.effective 会统一屏蔽，这里只管显示。"""
        book = self.var_book.get()
        self.var_book_label.set(core.BOOK_TYPES.get(book, core.BOOK_TYPES["text"]))
        for check in self.option_checks.values():
            check.pack_forget()
        for name in core.BOOK_OPTIONS.get(book, core.BOOK_OPTIONS["text"]):
            self.option_checks[name].pack(anchor="w", **self.option_pad)
        if self.infos is not None and self.analysis_key is not None:
            self.show_recommendation()      # 漫画多一项「纸面找平」的建议，文字书没有
        self.schedule_preview()

    def show_recommendation(self):
        try:                                # 抽查原文件的压缩方式（只读十几张图的文件头，很快）
            with fitz.open(self.var_input.get()) as doc:
                profile = core.source_profile(doc, self.infos)
        except Exception:
            profile = None
        rec = core.format_recommendation(self.infos, profile, self.analysis_key[1], book=self.var_book.get())
        self.rec = {name: rec[name] for name in ("max_angle", "min_angle", "quality", "dpi", "flatten")}
        for line in rec["lines"]:
            self.write_log(line)
        self.update_recommend_label()

    # (参数名, 名称, 单位, 是否「安静」)。安静的参数只在有事可做时才显示提示——
    # 绝大多数书的倾斜检测范围和渲染 DPI 用默认值就对，不值得每次都占一行
    REC_ROWS = (("max_angle", "倾斜检测范围", "°", True), ("min_angle", "不旋转角度", "°", False),
                ("quality", "JPEG 质量", "", False), ("dpi", "渲染 DPI", "", True),
                ("flatten", "纸面找平", "", True))      # 开关类：建议值是 True/False

    def recommendation_rows(self):
        """[(参数名, 名称, 单位, 安静, 界面变量, 当前值, 建议值, 默认值), ...]，只含本书有建议值的参数。"""
        opts, defaults = self.get_options(), core.Options()
        rows = []
        for name, title, unit, quiet in self.REC_ROWS:
            if self.rec.get(name) is not None:
                rows.append((name, title, unit, quiet, getattr(self, "var_" + name),
                             getattr(opts, name), self.rec[name], getattr(defaults, name)))
        return rows

    @staticmethod
    def rec_same(a, b):
        return a == b if isinstance(a, bool) else abs(a - b) < 1e-9

    @staticmethod
    def rec_text(value, unit):
        return ("开启" if value else "关闭") if isinstance(value, bool) else f"{value:g}{unit}"

    def update_recommend_label(self):
        """建议值的提示和按钮要跟着当前设置值走：用户手动改了数值，它们也得变。

        只要有一项还不是建议值，按钮就是「采用建议值」（一次全部采用）；全部都已经是建议值时
        变成「恢复默认值」（一次全部恢复）。建议值恰好全等于默认值时两个动作没有区别，不显示按钮。
        """
        for label in self.rec_labels.values():
            label.grid_forget()
        rows = self.recommendation_rows()
        line = 4
        for name, title, unit, quiet, _var, current, recommended, default in rows:
            adopted = self.rec_same(current, recommended)
            if quiet and adopted and self.rec_same(recommended, default):
                continue
            shown = self.rec_text(recommended, unit)
            text = (f"{title}: 当前值 {shown} 即为建议值" if adopted
                    else f"{title}: 建议 {shown}（理由见下方日志）")
            self.rec_labels[name].configure(text=text)
            self.rec_labels[name].grid(row=line, column=0, columnspan=2, sticky="w", padx=6, pady=3)
            line += 1
        all_adopted = all(self.rec_same(r[5], r[6]) for r in rows)
        same_as_default = all(self.rec_same(r[6], r[7]) for r in rows)
        if not rows or (all_adopted and same_as_default):
            self.btn_recommend.grid_forget()
        else:
            self.btn_recommend.configure(text="恢复默认值" if all_adopted else "采用建议值")
            self.btn_recommend.grid(row=self.rec_button_row, column=0, columnspan=2, sticky="ew",
                                    padx=6, pady=(0, 6))

    def toggle_recommendation(self):
        rows = self.recommendation_rows()
        all_adopted = all(self.rec_same(r[5], r[6]) for r in rows)
        for _name, _title, _unit, _quiet, var, _current, recommended, default in rows:
            value = default if all_adopted else recommended
            var.set(int(value) if isinstance(var, tk.IntVar) else value)
        # 倾斜检测范围、渲染 DPI 变了，原来的分析结果就不算数了：直接重新分析，不用等到点「开始处理」
        if self.analysis_key and self.analysis_key != self.current_key(self.get_options()):
            self.reanalyze()
        else:
            self.schedule_preview()

    # ------------------------------------------------------------ 进度的保存与恢复

    OPTION_VARS = ("book", "deskew", "center", "per_page", "clean", "upscale", "enhance", "flatten",
                   "max_angle", "min_angle", "quality", "dpi")

    def save_state(self):
        """把当前的选项、输出路径、看到第几页写进数据库。改动很小，随改随存。"""
        if not (self.store and self.book):
            return
        options = {}
        for name in self.OPTION_VARS:
            try:
                options[name] = getattr(self, "var_" + name).get()
            except tk.TclError:             # 输入框里暂时是空的或半截数字
                pass
        try:
            page = self.var_page.get()
        except tk.TclError:
            page = 1
        self.store.save_book(self.book["id"], options=options, output_path=self.var_output.get().strip(),
                             page_range=self.var_pages.get().strip(), current_page=page)

    def reset_options(self):
        """把「修正内容」的勾选和「参数」的数值全部恢复成默认值（Options 的默认值）。"""
        defaults = core.Options()
        for name in self.OPTION_VARS:
            field = {"clean": "clean_margin"}.get(name, name)      # 界面变量名和 Options 字段名只有这一处不同
            getattr(self, "var_" + name).set(getattr(defaults, field))

    def restore_state(self, book):
        for name, value in Store.options_of(book).items():
            if name in self.OPTION_VARS and (name != "book" or value in core.BOOK_TYPES):
                getattr(self, "var_" + name).set(value)
        # 以前保存的如果只是旧版的默认文件名（_reshaped），就换成现在的默认名；用户自己起的名字不动
        old_default = Path(book["path"]).with_name(Path(book["path"]).stem + "_reshaped.pdf")
        if book["output_path"] and Path(book["output_path"]) != old_default:
            self.var_output.set(book["output_path"])
        self.var_pages.set(book["page_range"] or "")
        self.var_page.set(min(max(book["current_page"] or 1, 1), self.page_count))
        self.skipped = self.store.flagged_pages(book["id"], "skip")
        self.deleted = self.store.flagged_pages(book["id"], "deleted")
        self.adjust = self.store.box_adjusts(book["id"])
        self.align = self.store.box_adjusts(book["id"], "box_align")
        self.cleanup = self.store.flagged_pages(book["id"], "cleanup")

    def update_skip_label(self):
        parts = []
        for title, marked in (("不修正", self.skipped), ("删除", self.deleted), ("去污", self.cleanup)):
            if marked:
                pages = sorted(i + 1 for i in marked)
                shown = ", ".join(map(str, pages[:6])) + (" …" if len(pages) > 6 else "")
                parts.append(f"{title} {len(pages)} 页: {shown}")
        self.lbl_skipped.configure(text="   ".join(parts))

    def update_skip_button(self, skip):
        self.var_skip.set(skip)
        self.btn_skip.configure(text="恢复本页修正" if skip else "本页不修正")

    # ------------------------------------------------------------ 手动微调版心边框

    def current_index(self):
        try:
            return min(max(self.var_page.get(), 1), self.page_count) - 1
        except tk.TclError:
            return 0

    def attach_adjust(self):
        """把逐页的手动设定挂到标准版心的统计结果上；core 的 page_box / cleanup_box 会自动用上。"""
        if self.ref is not None:
            self.ref["adjust"] = self.adjust
            self.ref["align"] = self.align
            self.ref["cleanup"] = self.cleanup

    def neighbor_index(self, direction):
        """前（-1）/后（+1）方向上最近的一个有版心、没被删除的页；没有就返回 None。"""
        i = self.current_index() + direction
        while self.infos and 0 <= i < len(self.infos):
            if self.infos[i].bbox and i not in self.deleted:
                return i
            i += direction
        return None

    def copy_neighbor_box(self, direction):
        index, other = self.current_index(), self.neighbor_index(direction)
        if self.ref is None or other is None or not self.infos[index].bbox:
            return
        self.set_adjust(index, core.box_like_neighbor(self.infos[index], self.infos[other], self.ref))

    def toggle_cleanup(self):
        if not self.page_count:
            return
        index = self.current_index()
        on = index not in self.cleanup
        (self.cleanup.add if on else self.cleanup.discard)(index)
        if self.store and self.book:
            self.store.set_page_flag(self.book["id"], index, "cleanup", on)
        self.update_skip_label()
        self.update_nudge_buttons()
        self.request_preview()

    def adjust_ready(self):
        """现在能不能调版心边框：全书分析好了（分析参数没改过），而且没在分析、处理。"""
        return (not self.locked and self.ref is not None
                and self.analysis_key == self.current_key(self.get_options()))

    def update_nudge_buttons(self):
        """选「上/下」时只有上下箭头可用，选「左/右」时只有左右箭头可用；全书分析完之前都不可用。"""
        ready = self.adjust_ready()
        vertical_edge = self.var_edge.get() in ("上", "下")
        for arrow, btn in self.nudge_buttons.items():
            usable = ready and (arrow in "↑↓") == vertical_edge
            btn.configure(state="normal" if usable else "disabled")
        index = self.current_index()
        has_box = ready and bool(self.infos) and index < len(self.infos) and bool(self.infos[index].bbox)
        self.btn_like_prev.configure(
            state="normal" if has_box and self.neighbor_index(-1) is not None else "disabled")
        self.btn_like_next.configure(
            state="normal" if has_box and self.neighbor_index(1) is not None else "disabled")
        self.btn_cleanup.configure(state="normal" if has_box else "disabled",
                                   text="取消去除污染" if index in self.cleanup else "去除边缘污染")
        offsets, aligned = self.adjust.get(index), self.align.get(index)
        self.btn_nudge_reset.configure(state="normal" if ready and (offsets or aligned) else "disabled")
        pending = offsets != aligned         # 红框和上次「版心居中」时的不一样：页面还没有按现在的红框对齐
        self.btn_center.configure(state="normal" if has_box and pending else "disabled")
        parts = [f"{name}{offsets[i] * 100:+.1f}%" for name, i in EDGES.items() if abs(offsets[i]) > 1e-9] if offsets else []
        text = "本页已微调: " + "  ".join(parts) if parts else ""
        if pending:
            text += ("  " if text else "红框已复原  ") + "（点「版心居中」按现在的红框对齐）"
        self.lbl_nudge.configure(text=text)

    def nudge(self, direction):
        """把选中的那条边移动一步。direction: -1 向上/向左，+1 向下/向右。"""
        if self.ref is None or not self.page_count:
            return
        index = self.current_index()
        offsets = list(self.adjust.get(index, (0.0, 0.0, 0.0, 0.0)))
        offsets[EDGES[self.var_edge.get()]] += direction * NUDGE_STEP
        offsets = tuple(0.0 if abs(o) < 1e-9 else round(o, 5) for o in offsets)
        self.set_adjust(index, offsets)

    def reset_nudge(self):
        """撤销这一页红框的调整，连同手动的「版心居中」：回到自动判断的结果。"""
        self.set_align(self.current_index(), None)
        self.set_adjust(self.current_index(), None)

    def center_by_box(self):
        """按现在的红框重新对齐这一页。记下的是这一刻的红框：之后再调红框页面不会动，要再点一次。"""
        if not self.adjust_ready() or not self.page_count:
            return
        index = self.current_index()
        self.set_align(index, self.adjust.get(index))
        self.update_nudge_buttons()
        self.hold_estimate()
        self.schedule_preview()

    def set_align(self, index, offsets):
        if offsets and any(offsets):
            self.align[index] = offsets
        else:
            self.align.pop(index, None)
        if self.store and self.book:
            self.store.set_box_adjust(self.book["id"], index, offsets, column="box_align")

    def set_adjust(self, index, offsets):
        if offsets and any(offsets):
            self.adjust[index] = offsets
        else:
            self.adjust.pop(index, None)
        if self.store and self.book:
            self.store.set_box_adjust(self.book["id"], index, offsets)
        self.update_nudge_buttons()
        self.hold_estimate()
        self.schedule_preview()              # 重新计算这一页的对齐并刷新显示（连按时合并成一次）

    def hold_estimate(self):
        """调版心边框往往要连着点很多下。预估输出大小要抽十几页真的处理一遍，每点一下就算一遍会把
        程序拖慢：所以点的时候先把排队中的、正在算的都停掉，等停手 2 秒之后再算（见 schedule_estimate）。"""
        self.nudged_at = time.monotonic()
        if self.estimate_after_id:
            self.after_cancel(self.estimate_after_id)
            self.estimate_after_id = None
        self.estimate_gen += 1               # 正在算的那一次看到序号变了就会中止
        self.estimate_key, self.estimate_running = None, False
        if self.lbl_estimate.cget("text"):
            self.lbl_estimate.configure(text="预计输出大小：调整结束后重新计算…")

    def update_delete_button(self, deleted):
        self.btn_delete.configure(text="恢复当前页" if deleted else "删除当前页",
                                  style="Restore.TButton" if deleted else "Delete.TButton")

    def toggle_delete(self):
        if not self.page_count:
            return
        index = min(max(self.var_page.get(), 1), self.page_count) - 1
        deleted = index not in self.deleted
        (self.deleted.add if deleted else self.deleted.discard)(index)
        if self.store and self.book:
            self.store.set_page_flag(self.book["id"], index, "deleted", deleted)
        self.update_skip_label()
        self.request_preview()

    def toggle_skip(self):
        if not self.page_count:
            return
        index = min(max(self.var_page.get(), 1), self.page_count) - 1
        skip = index not in self.skipped
        (self.skipped.add if skip else self.skipped.discard)(index)
        if self.store and self.book:
            self.store.set_page_flag(self.book["id"], index, "skip", skip)
        self.update_skip_label()
        self.request_preview()

    def on_close(self):
        self.save_state()
        self.destroy()

    def show_history(self):
        if not self.store:
            return
        win = tk.Toplevel(self)
        win.title("历史记录")
        win.geometry("900x380")
        win.transient(self)
        cols = [("name", "文件名", 260), ("pages", "页数", 50), ("skipped", "不修正", 60), ("deleted", "删除", 50),
                ("state", "状态", 110), ("updated", "最后修改", 140), ("path", "路径", 400)]
        tree = ttk.Treeview(win, columns=[c[0] for c in cols], show="headings", selectmode="browse")
        for key, title, width in cols:
            tree.heading(key, text=title)
            tree.column(key, width=width, anchor="w" if key in ("name", "path") else "center")
        tree.pack(fill="both", expand=True, padx=8, pady=8)

        def reload():
            tree.delete(*tree.get_children())
            for b in self.store.list_books():
                state = "已输出" if b["processed_at"] else ("已分析" if b["analyzed"] else "未分析")
                if not Path(b["path"]).exists():
                    state += "（文件不在）"
                tree.insert("", "end", iid=str(b["id"]), values=(
                    Path(b["path"]).name, b["page_count"], b["skipped"] or "", b["deleted"] or "", state,
                    b["updated_at"].replace("T", " "), b["path"]))

        def selected():
            sel = tree.selection()
            return (int(sel[0]), tree.set(sel[0], "path")) if sel else (None, None)

        def open_selected(_event=None):
            book_id, path = selected()
            if not path:
                return
            if not Path(path).exists():
                messagebox.showwarning("文件不在", path + "  文件已被移动或改名。请用「打开…」重新选择它——"
                                       "记录是按文件内容匹配的，调整进度会自动接上。", parent=win)
                return
            win.destroy()
            self.open_file(path)

        def delete_selected():
            book_id, path = selected()
            if book_id is None:
                return
            if not messagebox.askyesno("删除记录", Path(path).name + "  删除这本书保存的调整进度？（不会删除 PDF 文件）",
                                       parent=win):
                return
            self.store.delete_book(book_id)
            if self.book and self.book["id"] == book_id:
                self.book = None            # 当前这本的记录没了：之后的改动不再保存，直到重新打开
            reload()

        bar = ttk.Frame(win)
        bar.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Button(bar, text="打开", command=open_selected).pack(side="left")
        ttk.Button(bar, text="删除记录", command=delete_selected).pack(side="left", padx=6)
        ttk.Button(bar, text="关闭", command=win.destroy).pack(side="right")
        ttk.Label(win, text=f"数据库: {self.store.path}", foreground="#666").pack(anchor="w", padx=8, pady=(0, 8))
        tree.bind("<Double-1>", open_selected)
        reload()
        return win

    # ------------------------------------------------------------ 输出大小预估

    def schedule_estimate(self):
        """相关设置变了就重新预估输出大小。翻页不影响大小，不会触发重算。"""
        opts = self.get_options()
        if self.locked:
            return                          # 处理期间不预估：设置改不了，也免得和处理抢 CPU
        if self.analysis_key != self.current_key(opts):
            return                          # 全书分析还没好（或者分析参数改了），无从预估
        try:
            indices = core.parse_pages(self.var_pages.get().strip(), self.page_count)
        except ValueError:
            return
        key = (self.analysis_key, opts.deskew, opts.center, opts.per_page, opts.clean_margin, opts.upscale, opts.enhance,
               opts.flatten, opts.min_angle, opts.quality, frozenset(self.skipped), frozenset(self.deleted), tuple(indices),
               tuple(sorted(self.adjust.items())), tuple(sorted(self.align.items())), frozenset(self.cleanup))
        if key == self.estimate_key or not indices:
            return
        self.estimate_key = key
        if self.estimate_after_id:
            self.after_cancel(self.estimate_after_id)
        nudging = time.monotonic() - self.nudged_at < NUDGE_ESTIMATE_DELAY / 1000
        delay = NUDGE_ESTIMATE_DELAY if nudging else ESTIMATE_DELAY
        self.estimate_after_id = self.after(delay, lambda: self.start_estimate(opts, indices))

    def start_estimate(self, opts, indices):
        self.estimate_after_id = None
        if self.infos is None or self.analysis_key != self.current_key(opts):
            self.estimate_key = None        # 排队期间分析结果被清掉了（重新分析、换了一本书）：作废，等下次再算
            return
        self.estimate_gen += 1
        self.estimate_running = True
        self.lbl_estimate.configure(text="预计输出大小：计算中…")
        infos = [self.infos[i] for i in indices]
        threading.Thread(target=self._estimate_thread, daemon=True,
                         args=(self.estimate_gen, self.var_input.get(), infos, self.ref, opts,
                               frozenset(self.skipped), frozenset(self.deleted))).start()

    def _estimate_thread(self, gen, src, infos, ref, opts, skipped, deleted):
        try:
            with fitz.open(src) as doc:
                result = core.estimate_output_size(doc, infos, ref, opts, skipped, deleted,
                                                   cache=self.estimate_cache,
                                                   cancelled=lambda: gen != self.estimate_gen)
            self.msgs.put(("estimate", gen, result, Path(src).stat().st_size))
        except core.Cancelled:
            pass
        except Exception as e:
            self.msgs.put(("estimate", gen, None, f"{type(e).__name__}: {e}"))

    def current_key(self, opts):
        return (self.var_input.get(), opts.max_angle, opts.dpi)

    # ------------------------------------------------------------ 文件

    def browse_input(self):
        path = filedialog.askopenfilename(title="选择扫描 PDF", filetypes=[("PDF", "*.pdf")])
        if path:
            self.open_file(path)

    def browse_output(self):
        initial = Path(self.var_output.get() or "output.pdf")
        path = filedialog.asksaveasfilename(title="输出文件", defaultextension=".pdf",
                                            initialdir=initial.parent, initialfile=initial.name,
                                            filetypes=[("PDF", "*.pdf")])
        if path:
            self.var_output.set(str(Path(path)))

    def open_file(self, path):
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("处理中", "请先等待当前任务结束，或点「取消」。")
            return
        path = Path(path)
        try:
            with fitz.open(path) as doc:
                if doc.needs_pass:
                    raise ValueError("该 PDF 有密码保护")
                self.page_count = doc.page_count
        except Exception as e:
            messagebox.showerror("无法打开", f"{path}\n\n{e}")
            return
        self.save_state()                   # 先把上一本的进度存好
        self.reset_options()                # 每本书从默认设置开始，不沿用上一本的；有保存记录的随后再恢复
        self.var_input.set(str(path))
        self.var_output.set(str(core.default_output_path(path)))
        self.infos = self.ref = self.analysis_key = None
        self.preview_data = None
        self.rec = {}
        self.update_recommend_label()
        self.estimate_gen += 1              # 作废上一本书还在算的预估
        self.estimate_key, self.estimate_cache, self.estimate_running = None, {}, False
        self.lbl_estimate.configure(text="")
        self.spin_page.configure(to=self.page_count)
        self.var_page.set(1)
        self.var_pages.set("")
        self.skipped = set()
        self.deleted = set()
        self.adjust = {}
        self.align = {}
        self.cleanup = set()
        self.btn_delete.configure(state="normal")
        self.lbl_total.configure(text=f"/ {self.page_count}")
        self.btn_run.configure(state="normal")
        self.btn_skip.configure(state="normal")
        self.write_log(f"打开 {path.name}（{self.page_count} 页）", clear=True)

        self.book, infos = None, None
        if self.store:
            self.book, created = self.store.open_book(path, self.page_count)
            if not created:
                self.restore_state(self.book)
                opts = self.get_options()
                infos = self.store.load_analysis(self.book, (opts.max_angle, opts.dpi))
                self.write_log(f"已恢复上次的进度（{self.book['updated_at'].replace('T', ' ')}）：选项、"
                               f"第 {self.var_page.get()} 页、不修正 {len(self.skipped)} 页、删除 {len(self.deleted)} 页、"
                               f"微调版心 {len(self.adjust)} 页"
                               + ("、全书分析结果。" if infos else "。分析结果需要重新生成。"))
        self.update_skip_label()
        self.btn_reanalyze.pack(fill="x", pady=(6, 0), before=self.btn_run)
        if infos:
            self.infos, self.ref = infos, core.compute_reference(infos)
            self.attach_adjust()
            self.analysis_key = self.current_key(self.get_options())
            self.progress.configure(value=100)
            self.lbl_status.configure(text="已恢复上次的进度。可以继续翻页调整，确认后点「开始处理」。")
            self.show_recommendation()
        else:
            # 不自动分析（用户要求）：先选好书的类型、设好参数，再点「分析全书」；直接点「开始处理」也会先分析
            self.progress.configure(value=0)
            self.lbl_status.configure(text="请选择书的类型、设好参数，然后点「分析全书」（或直接点「开始处理」）。")
        self.update_analyze_button()
        self.schedule_preview()

    def update_analyze_button(self):
        analyzed = self.infos is not None and self.analysis_key == self.current_key(self.get_options())
        self.btn_reanalyze.configure(text="重新分析" if analyzed else "分析全书")

    # ------------------------------------------------------------ 分析 / 处理（工作线程）

    def start_worker(self, target, *args):
        self.cancel_event.clear()
        self.btn_cancel.configure(state="normal")
        self.worker = threading.Thread(target=self._guard, args=(target, *args), daemon=True)
        self.worker.start()

    def _guard(self, target, *args):
        try:
            target(*args)
        except core.Cancelled:
            self.msgs.put(("cancelled",))
        except Exception as e:
            self.msgs.put(("error", f"{type(e).__name__}: {e}"))

    def reanalyze(self):
        """丢掉从历史记录里恢复的分析结果，重新分析全书。手动标记（不修正、删除）和各项设置都保留。"""
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("处理中", "请先等待当前任务结束，或点「取消」。")
            return
        self.infos = self.ref = self.analysis_key = None
        self.rec = {}
        self.update_recommend_label()
        self.estimate_gen += 1
        self.estimate_key, self.estimate_cache, self.estimate_running = None, {}, False
        self.lbl_estimate.configure(text="")
        self.progress.configure(value=0)
        self.write_log("分析全书…")
        self.start_analysis()
        self.schedule_preview()

    def start_analysis(self):
        opts = self.get_options()
        self.analyzing_key = self.current_key(opts)
        self.set_locked(True)
        self.start_worker(self._analyze_thread, self.analyzing_key, opts)

    def _analyze_thread(self, key, opts):
        with fitz.open(key[0]) as doc:
            infos = core.analyze_document(
                doc, range(doc.page_count), opts,
                progress=lambda k, n: self.msgs.put(("progress", "分析", k, n)),
                cancelled=self.cancel_event.is_set)
        self.msgs.put(("analyzed", key, infos))

    def run(self):
        opts = self.get_options()
        src, out = self.var_input.get(), self.var_output.get().strip()
        if not out:
            messagebox.showwarning("输出文件", "请指定输出文件。")
            return
        if Path(out).resolve() == Path(src).resolve():
            messagebox.showwarning("输出文件", "输出文件不能和输入文件相同。")
            return
        try:
            indices = core.parse_pages(self.var_pages.get().strip(), self.page_count)
            if not indices:
                raise ValueError
        except ValueError:
            messagebox.showwarning("处理范围", "页码范围的写法不正确。例: 1-20 或 3,5,8-12")
            return
        if all(i in self.deleted for i in indices):
            messagebox.showwarning("处理范围", "要处理的页全部被标记为删除，没有可输出的页。")
            return
        if Path(out).exists() and not messagebox.askyesno("覆盖确认", f"{out}\n\n文件已存在，要覆盖吗？"):
            return

        job = (opts, indices, out, not self.var_pages.get().strip())
        self.set_locked(True)               # 从这一刻起设置就定下来了（要先重新分析的也一样）
        if self.analysis_key == self.current_key(opts):
            self.start_process(job)
        elif self.worker and self.worker.is_alive() and self.analyzing_key == self.current_key(opts):
            self.pending_run = job          # 分析还没跑完：跑完后自动接着处理
            self.lbl_status.configure(text="等待分析结束后开始处理…")
        else:                                # 检测范围或 DPI 改过：重新分析
            self.pending_run = job
            if self.worker and self.worker.is_alive():
                self.cancel_event.set()
                self.worker.join()
                self._drain_stale()
            self.start_analysis()

    def _drain_stale(self):
        while not self.msgs.empty():
            self.msgs.get_nowait()

    def start_process(self, job):
        opts, indices, out, whole = job
        infos = [self.infos[i] for i in indices]
        self.save_state()
        # 标准版心和挂在上面的逐页设定交副本：处理线程一页一页地读它们，不能和界面共用
        ref = dict(self.ref, adjust=dict(self.adjust), align=dict(self.align), cleanup=frozenset(self.cleanup))
        self.start_worker(self._process_thread, self.var_input.get(), infos, ref, opts, out, whole,
                          frozenset(self.skipped), frozenset(self.deleted))

    def _process_thread(self, src, infos, ref, opts, out, whole, skipped, deleted):
        with fitz.open(src) as doc:
            changed = core.process_document(
                doc, infos, opts, out, ref=ref, copy_toc=whole, skip_pages=skipped, delete_pages=deleted,
                progress=lambda k, n: self.msgs.put(("progress", "处理", k, n)),
                log=lambda s: self.msgs.put(("log", s)),
                cancelled=self.cancel_event.is_set)
        self.msgs.put(("done", changed, len(infos), sum(1 for p in infos if p.index in deleted), out))

    def set_locked(self, locked):
        """分析、处理期间锁住所有会改变输出的控件，结束（完成、取消、出错）后恢复。

        处理用的是点「开始处理」那一刻的设置：期间再改，界面上看着生效了，输出里却没有。分析期间
        也锁（用户要求）：打开文件之后等全书分析完才能动设置。
        「取消」、翻页和预览、放大镜、辅助线不锁——一本书要几分钟，边等边翻看是有用的，而且都是只读的。
        """
        self.locked = locked
        for widget in self.lockable:
            # 下拉框平时是「只读」（只能选、不能打字），恢复成 normal 就能打字了
            widget.configure(state="disabled" if locked else "readonly" if isinstance(widget, ttk.Combobox) else "normal")
        self.btn_reanalyze.configure(state="disabled" if locked else "normal")
        for btn in (self.btn_run, self.btn_skip, self.btn_delete):
            btn.configure(state="normal" if self.page_count and not locked else "disabled")
        self.update_nudge_buttons()
        self.draw_preview()                 # 红框上拖动用的小方块跟着出现、消失
        if locked:                          # 排队中的、正在算的预估都停掉，解锁之后再算
            if self.estimate_after_id:
                self.after_cancel(self.estimate_after_id)
                self.estimate_after_id, self.estimate_key = None, None
            if self.estimate_running:
                self.estimate_gen += 1
                self.estimate_key, self.estimate_running = None, False
                self.lbl_estimate.configure(text="预计输出大小：处理结束后重新计算…")
        else:
            self.schedule_estimate()

    def cancel(self):
        self.pending_run = None
        self.cancel_event.set()

    # ------------------------------------------------------------ 预览（独立线程）

    def on_page_key(self, event, delta):
        # 焦点在输入框里时，方向键和 Home/End 是用来改数值、移动光标的，不能拿来翻页；
        # PageUp/PageDown 在输入框里没有别的用途，始终翻页
        typing = event.widget.winfo_class() in ("TSpinbox", "TEntry", "Entry", "Text", "TCombobox")
        if typing and event.keysym not in ("Prior", "Next"):
            return
        self.step_page(delta)

    def step_page(self, delta):
        if not self.page_count:
            return
        try:
            page = self.var_page.get()
        except tk.TclError:
            page = 1
        self.var_page.set(min(max(page + delta, 1), self.page_count))
        self.schedule_preview()

    def schedule_preview(self):
        if self.preview_after_id:
            self.after_cancel(self.preview_after_id)
        self.preview_after_id = self.after(150, self.request_preview)

    def request_preview(self):
        self.preview_after_id = None
        if not self.page_count:
            return
        try:
            index = min(max(self.var_page.get(), 1), self.page_count) - 1
        except tk.TclError:
            return
        opts = self.get_options()
        self.preview_gen += 1
        analyzed = self.analysis_key == self.current_key(opts)
        info = self.infos[index] if analyzed else None
        ref = self.ref if analyzed else None
        skip = index in self.skipped
        deleted = index in self.deleted
        self.update_skip_button(skip)
        self.update_delete_button(deleted)
        self.update_nudge_buttons()
        self.update_recommend_label()
        self.save_state()                   # 翻页、改选项都会走到这里
        self.schedule_estimate()
        threading.Thread(target=self._preview_thread, daemon=True,
                         args=(self.preview_gen, self.var_input.get(), index, info, ref, opts, skip,
                               deleted)).start()

    def _preview_thread(self, gen, src, index, info, ref, opts, skip, deleted):
        try:
            with fitz.open(src) as doc:
                provisional = info is None
                if provisional:                # 全书分析还没结束：先单独分析这一页
                    info = core.analyze_page(doc, index, opts)
                before, after, box, status = core.preview_page(doc, info, ref, opts, skip)
                shift = core.plan_page(info, ref, opts, skip)[1]
            if deleted:
                after, box, status = before, None, "本页已删除，不会输出到新 PDF"
            elif provisional and not opts.per_page and not skip:
                status += "  ※全书分析完成前为暂定结果"
            full = (before, after, (0.0, 0.0) if deleted else shift)     # 原尺寸的图：放大镜用

            def fit(img):
                scale = PREVIEW_MAX_SIDE / max(img.shape[:2])
                return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img
            self.msgs.put(("preview", gen, fit(before), fit(after), box, status, deleted, full, index))
        except Exception as e:
            self.msgs.put(("preview_error", gen, f"{type(e).__name__}: {e}"))

    def draw_preview(self):
        if not self.preview_data:
            return
        before, after, box, deleted = self.preview_data
        for slot, (cv, img) in enumerate(zip(self.canvases, (before, after))):
            cw, ch = cv.winfo_width(), cv.winfo_height()
            if cw < 20 or ch < 20:
                continue
            h, w = img.shape[:2]
            s = min((cw - 8) / w, (ch - 8) / h)
            dw, dh = max(int(w * s), 1), max(int(h * s), 1)
            small = cv2.resize(img, (dw, dh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
            ok, png = cv2.imencode(".png", small)
            self.photos[slot] = tk.PhotoImage(data=base64.b64encode(png.tobytes()))
            x0, y0 = (cw - dw) // 2, (ch - dh) // 2
            self.preview_geometry[slot] = (x0, y0, dw, dh)
            cv.delete("all")
            cv.create_image(x0, y0, anchor="nw", image=self.photos[slot])
            if self.var_guides.get():
                # 页面中心十字线；处理后一侧再画出版心框
                cv.create_line(x0 + dw / 2, y0, x0 + dw / 2, y0 + dh, fill="#2a7fff", dash=(4, 4))
                cv.create_line(x0, y0 + dh / 2, x0 + dw, y0 + dh / 2, fill="#2a7fff", dash=(4, 4))
                if slot == 1 and box:
                    self.draw_box(box)
            if slot == 1 and deleted:       # 已删除的页：在「处理后」一侧打上红叉
                cv.create_line(x0, y0, x0 + dw, y0 + dh, fill="#d00000", width=4)
                cv.create_line(x0 + dw, y0, x0, y0 + dh, fill="#d00000", width=4)
                cv.create_rectangle(x0 + dw / 2 - 130, y0 + dh / 2 - 22, x0 + dw / 2 + 130, y0 + dh / 2 + 22,
                                    fill="#d00000", outline="")
                cv.create_text(x0 + dw / 2, y0 + dh / 2, text="已删除 · 不输出", fill="white",
                               font=("Microsoft YaHei UI", 14, "bold"))

    # ------------------------------------------------------------ 用鼠标拖版心框

    def box_editable(self):
        """预览里的红框现在能不能拖。画面上的红框必须就是当前页、当前设置下的：翻了页或改了设置、
        新的预览还没出来的那一小会儿不能拖——拖动量是相对画面上的红框算的。"""
        if not (self.preview_data and self.preview_geometry[1] and self.var_guides.get() and self.adjust_ready()):
            return False
        box, deleted = self.preview_data[2:]
        return (bool(box) and not deleted and self.preview_after_id is None
                and self.preview_shown_gen == self.preview_gen and self.preview_index == self.current_index())

    def handle_points(self, box):
        """八个小方块在「处理后」画布上的位置：[(小方块, x, y), ...]。"""
        x0, y0, dw, dh = self.preview_geometry[1]
        xs = {0: box[0], 2: box[2], None: (box[0] + box[2]) / 2}
        ys = {1: box[1], 3: box[3], None: (box[1] + box[3]) / 2}
        return [(h, x0 + xs[h[0]] * dw, y0 + ys[h[1]] * dh) for h in HANDLES]

    def handle_at(self, x, y):
        if not self.box_editable():
            return None
        near = [(max(abs(x - hx), abs(y - hy)), h) for h, hx, hy in self.handle_points(self.preview_data[2])]
        reach, handle = min(near, key=lambda item: item[0])
        return handle if reach <= HANDLE_REACH else None

    def draw_box(self, box):
        """在「处理后」一侧画红框；能拖的时候在四角和四边的中点画上白色小方块。"""
        cv = self.canvases[1]
        x0, y0, dw, dh = self.preview_geometry[1]
        cv.delete("box")
        cv.create_rectangle(x0 + box[0] * dw, y0 + box[1] * dh, x0 + box[2] * dw, y0 + box[3] * dh,
                            outline="#e03131", tags="box")
        if self.drag or self.box_editable():
            for _handle, hx, hy in self.handle_points(box):
                cv.create_rectangle(hx - HANDLE_HALF, hy - HANDLE_HALF, hx + HANDLE_HALF, hy + HANDLE_HALF,
                                    fill="white", outline="#e03131", tags=("box", "handle"))

    def update_cursor(self, x, y):
        handle = self.drag[0] if self.drag else self.handle_at(x, y)
        try:
            self.canvases[1].configure(cursor=HANDLE_CURSORS[handle] if handle else "")
        except tk.TclError:                 # 这几个光标名是 Windows 上的；别的平台没有就算了
            pass

    def on_canvas_press(self, slot, x, y):
        handle = self.handle_at(x, y) if slot == 1 else None
        if handle is None:
            return self.show_loupe(slot, x, y)
        self.drag = (handle, x, y, self.preview_data[2])
        self.drag_box = self.preview_data[2]

    def on_canvas_motion(self, slot, x, y):
        if not self.drag:
            return self.show_loupe(slot, x, y)
        (xi, yi), px, py, start = self.drag
        _x0, _y0, dw, dh = self.preview_geometry[1]
        shift = self.preview_full[2] if self.preview_full else (0.0, 0.0)
        box = list(start)
        for edge, moved, offset in ((xi, (x - px) / dw, shift[0]), (yi, (y - py) / dh, shift[1])):
            if edge is None:
                continue
            # 红框是画在平移之后的页面上的：换算回原来的位置不能出页面；也不能拖过对面那条边
            value = min(max(start[edge] + moved, offset), 1.0 + offset)
            opposite = start[(edge + 2) % 4]
            box[edge] = min(value, opposite - MIN_BOX) if edge < 2 else max(value, opposite + MIN_BOX)
        self.drag_box = tuple(box)
        self.draw_box(self.drag_box)

    def on_canvas_release(self, slot):
        if not self.drag:
            return self.hide_loupe()
        (xi, yi), _px, _py, start = self.drag
        box, self.drag, self.drag_box = self.drag_box, None, None
        index = self.current_index()
        if self.locked or self.ref is None or not self.infos or box == start:
            return self.draw_preview()
        # 记成和箭头微调一样的「四条边各自的移动量」：拖动的边按拖动量算，没动的边保持原来的
        info = self.infos[index]
        current = core.page_box(info, self.ref)
        base = self.ref["boxes"].get(index, info.bbox)
        offsets = list(self.adjust.get(index, (0.0, 0.0, 0.0, 0.0)))
        for edge in (xi, yi):
            if edge is not None:
                offsets[edge] = current[edge] + (box[edge] - start[edge]) - base[edge]
        offsets = tuple(0.0 if abs(o) < 1e-5 else round(o, 5) for o in offsets)
        # 新的预览出来之前先按拖到的位置画着，免得红框跳回去再跳过来
        self.preview_data = (*self.preview_data[:2], box, self.preview_data[3])
        self.set_adjust(index, offsets)     # 画面不用重画：拖的时候已经画成这样了，小方块也留着，等新的预览来换

    # ------------------------------------------------------------ 放大镜

    def show_loupe(self, slot, x, y):
        """在两个预览画布上同时放大同一处内容。整页缩到窗口大小时，显示增强的效果是看不出来的。"""
        if not (self.preview_full and self.preview_geometry[slot]):
            return
        before, after, shift = self.preview_full
        x0, y0, dw, dh = self.preview_geometry[slot]
        u, v = (x - x0) / dw, (y - y0) / dh                 # 鼠标指着页面的什么位置（0~1）
        if not (0 <= u <= 1 and 0 <= v <= 1):
            return self.hide_loupe()
        # 处理后的内容挪动过：左边要往回找，两边才是同一处（旋转的那一点差别忽略）
        if slot == 1:
            spots = ((u - shift[0], v - shift[1]), (u, v))
        else:
            spots = ((u, v), (u + shift[0], v + shift[1]))
        for k, (img, (pu, pv)) in enumerate(zip((before, after), spots)):
            h, w = img.shape[:2]
            half = max(int(LOUPE_SPAN * w / 2), 8)
            cx, cy = int(np.clip(pu * w, half, w - half)), int(np.clip(pv * h, half, h - half))
            crop = img[cy - half:cy + half, cx - half:cx + half]
            crop = cv2.resize(crop, (LOUPE_SIZE, LOUPE_SIZE),
                              interpolation=cv2.INTER_NEAREST if 2 * half < LOUPE_SIZE else cv2.INTER_AREA)
            ok, png = cv2.imencode(".png", crop)
            self.loupe_photos[k] = tk.PhotoImage(data=base64.b64encode(png.tobytes()))
            gx, gy, gw, gh = self.preview_geometry[k]
            px = int(np.clip(gx + spots[k][0] * gw - LOUPE_SIZE / 2 if k != slot else x - LOUPE_SIZE / 2,
                             0, max(self.canvases[k].winfo_width() - LOUPE_SIZE, 0)))
            py = int(np.clip(gy + spots[k][1] * gh - LOUPE_SIZE / 2 if k != slot else y - LOUPE_SIZE / 2,
                             0, max(self.canvases[k].winfo_height() - LOUPE_SIZE, 0)))
            cv = self.canvases[k]
            cv.delete("loupe")
            cv.create_image(px, py, anchor="nw", image=self.loupe_photos[k], tags="loupe")
            cv.create_rectangle(px, py, px + LOUPE_SIZE, py + LOUPE_SIZE, outline="#2a7fff", width=2, tags="loupe")

    def hide_loupe(self):
        for cv in self.canvases:
            cv.delete("loupe")

    # ------------------------------------------------------------ 消息处理

    def write_log(self, text, clear=False):
        self.log.configure(state="normal")
        if clear:
            self.log.delete("1.0", "end")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _idle(self):
        self.set_locked(False)
        self.btn_cancel.configure(state="disabled")
        self.update_analyze_button()

    def _poll(self):
        try:
            while True:
                self._handle(self.msgs.get_nowait())
        except queue.Empty:
            pass
        self.after(50, self._poll)

    def _handle(self, msg):
        kind = msg[0]
        if kind == "progress":
            _, phase, k, n = msg
            self.progress.configure(value=100 * k / n)
            self.lbl_status.configure(text=f"{phase}中… {k}/{n}")
        elif kind == "log":
            self.write_log(msg[1])
        elif kind == "analyzed":
            _, key, infos = msg
            self.infos, self.analysis_key = infos, key
            if self.store and self.book:
                self.store.save_analysis(self.book["id"], infos, key[1:])
            self.ref = core.compute_reference(infos)
            self.attach_adjust()
            scans = sum(1 for p in infos if p.bbox)
            self.write_log(f"分析完成：{len(infos)} 页中有 {scans} 页可修正。")
            self.show_recommendation()
            self.lbl_status.configure(text="分析完成。可以翻页预览效果，确认后点「开始处理」。")
            self.request_preview()
            if self.pending_run:
                job, self.pending_run = self.pending_run, None
                self.start_process(job)
            else:
                self._idle()
        elif kind == "done":
            _, changed, total, removed, out = msg
            summary = f"共 {total} 页，修正 {changed} 页" + (f"，删除 {removed} 页" if removed else "")
            self._idle()
            self.lbl_status.configure(text=f"完成：{summary}")
            if self.store and self.book:
                self.store.save_book(self.book["id"], processed_at=datetime.now().isoformat(timespec="seconds"))
            self.write_log(f"完成：{summary} → {out}（实际大小 {core.format_size(Path(out).stat().st_size)}）")
            messagebox.showinfo("完成", f"{summary}。\n\n{out}")
        elif kind == "cancelled":
            self._idle()
            self.progress.configure(value=0)
            self.lbl_status.configure(text="已取消")
            self.write_log("已取消。")
        elif kind == "error":
            self.pending_run = None
            self._idle()
            self.lbl_status.configure(text="出错")
            self.write_log("错误: " + msg[1])
            messagebox.showerror("出错", msg[1])
        elif kind == "preview":
            _, gen, before, after, box, status, deleted, full, index = msg
            if gen == self.preview_gen and not self.drag:       # 正拖着的时候不换图，松手后反正要重新预览
                self.preview_data = (before, after, box, deleted)
                self.preview_index, self.preview_shown_gen = index, gen
                self.preview_full = full
                self.lbl_page_status.configure(text=status)
                self.draw_preview()
        elif kind == "estimate":
            _, gen, result, extra = msg
            if gen == self.estimate_gen:
                self.estimate_running = False
                if result is None:
                    self.lbl_estimate.configure(text="预计输出大小：无法预估（" + extra + "）")
                else:
                    size, copied, encoded, sampled = result
                    self.lbl_estimate.configure(
                        text=f"预计输出约 {core.format_size(size)}（原文件 {core.format_size(extra)}）"
                             + (f"，按 {sampled} 页抽样推算" if encoded else ""))
        elif kind == "preview_error":
            if msg[1] == self.preview_gen:
                self.lbl_page_status.configure(text="预览失败: " + msg[2])


def main():
    try:  # Windows 高 DPI 屏幕下避免界面发虚
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    App(sys.argv[1] if len(sys.argv) > 1 else None).mainloop()


if __name__ == "__main__":
    main()
