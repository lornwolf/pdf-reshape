#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""扫描书籍 PDF 整形工具：纠正页面倾斜，并把版心（文字区域）移到页面正中。

用法:
    python pdf_reshape.py input.pdf                 # 输出 input（校正版）.pdf
    python pdf_reshape.py input.pdf -o out.pdf
    python pdf_reshape.py input.pdf --pages 1-20    # 只处理部分页（试效果用）
    python pdf_reshape.py input.pdf --clean-margin  # 顺便把版心外的黑边/阴影涂成纸色

处理分两遍:
    第 1 遍  分析每页的倾斜角度和版心位置，统计全书的标准版心尺寸
    第 2 遍  对每页做一次「旋转 + 平移」的仿射变换（只重采样一次），写入新 PDF
"""
import argparse
import bisect
import math
import struct
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import fitz  # PyMuPDF
import numpy as np

ANALYSIS_VERSION = 6        # 倾斜/版心检测的算法版本。改了检测逻辑就加 1，让已保存的分析结果失效
ANALYSIS_LONG_SIDE = 1200   # 分析用缩略图的长边像素数
FULL_PAGE_TOLERANCE = 0.02  # 版心尺寸与标准版心相差在此以内（或更大）才算「整页」，否则按「半页」处理
HANGING_TOLERANCE = 0.03    # 全部内容比标准的正文宽出这么多，才认为有东西（页码、页眉）挂在正文外面
OVERSIZE_TOLERANCE = 0.025  # 版心比标准版心大出这么多，才怀疑它是被空白处的杂质撑大的
EDGE_MATCH_TOLERANCE = 0.03  # 半页的版心边缘与同类页相差在此以内，才认为是同一条版心边
FLUSH_RATIO = 0.35          # 略窄的页：一边离标准版心的距离不到另一边的这个比例，就认为这一边是对齐的、缩进全在另一边
NARROW_TOLERANCE = 0.12     # 版心只比标准版心小这么多以内时，可以放心地单独居中（误差不超过它的一半）
FULL_BLEED_RATIO = 0.95     # 内容占满页面超过这个比例时，视为整页图片，不处理
COLOR_CHROMA = 20           # 页面里彩度（RGB 三通道的最大差）超过这个值的像素占 5% 以上，才算彩色页
PAPER_LIKE = 200            # 页边一圈里亮度至少这么高的像素才算纸；见 is_full_bleed_color


@dataclass
class PageInfo:
    index: int
    mode: str = "copy"          # copy: 原样复制 / native: 取内嵌原图 / render: 渲染成图
    xref: int = 0
    bilevel: bool = False       # 原图是否为黑白二值图
    dpi: float = 0.0            # 图像相对于页面尺寸的分辨率
    angle: float = 0.0          # 需要施加的旋转角度（度，逆时针为正）
    vertical: bool = False      # 是否竖排（倾斜检测时列方向的投影更有规律）
    bbox: tuple | None = None   # 旋转后的版心 (x0, y0, x1, y1)，按图像宽高归一化到 0~1
    core_bbox: tuple | None = None  # 只算成块内容（文字行、插图）的版心；见 effective_boxes
    outliers: list | None = None    # 伸出核心版心之外的小块 [(x0, y0, x1, y1), ...]：页码、序号，也可能是污渍
    dense_bbox: tuple | None = None  # 按墨迹密度求出的正文范围 (x0, y0, x1, y1)；见 dense_extent
    enhance: str = ""           # 「显示增强」对这一页适合用的方法，空 = 增强了也没有明显改善；见 assess_enhancement
    shade: float = 0.0          # 灰度/彩色页：页边的纸色比中部暗多少级（订口阴影）；见 measure_shade
    full_bleed_color: bool = False  # 彩色、画面一直铺到页边的页（封面等）：文字书里原样保留；见 is_full_bleed_color
    scan_rect: tuple = (0.0, 0.0, 1.0, 1.0)  # 扫描图在页面中的范围（归一化）
    # 以下两项每次 load_page_image 时重新填写，不依赖保存下来的分析结果
    jpeg_block: int = 0         # 原图是 JPEG 时，它的编码块大小（灰度 8、彩色 16）；不是 JPEG 为 0
    grid: tuple = (0, 0)        # 原图左上角在整页画布中的像素位置（JPEG 块网格的原点）
    note: str = ""


BOOK_TYPES = {"text": "文字书", "manga": "漫画书"}
# 各类书在界面上显示、在处理中生效的「修正内容」选项。不在列表里的选项对这类书一律视为关闭（Options.effective）
BOOK_OPTIONS = {"text": ("deskew", "center", "per_page", "clean_margin", "upscale", "enhance"),
                "manga": ("deskew", "center", "per_page", "clean_margin", "flatten")}


@dataclass
class Options:
    book: str = "text"          # 书的类型：text 文字书 / manga 漫画书。决定哪些选项可用、给哪些建议
    deskew: bool = True         # 倾斜校正
    center: bool = True         # 版心居中
    per_page: bool = False      # 每页各自居中（不参照全书标准版心）
    clean_margin: bool = False  # 把版心以外涂成纸色
    upscale: bool = False       # 黑白二值页旋转时以 2 倍分辨率输出（笔画边缘更平滑，体积约 2.7 倍）
    enhance: bool = False       # 显示增强：对能有明显改善的黑白页，自动选合适的方法增强显示效果
    flatten: bool = False       # 纸面找平（漫画）：把灰度/彩色页的纸面按当地纸色拉白，页边的灰影就没了
    max_angle: float = 5.0      # 倾斜检测范围 ±度
    min_angle: float = 0.1      # 小于此角度不旋转
    dpi: int = 300              # 无法直接取原图的页面的渲染 DPI
    quality: int = 90           # JPEG 质量

    def effective(self):
        """按书的类型把不适用的选项关掉之后的副本：文字书没有「纸面找平」，漫画没有「显示增强」等。
        界面上换了类型，藏起来的勾选框可能还勾着，所以一律在这里统一屏蔽，核心函数不用各自判断。"""
        allowed = BOOK_OPTIONS.get(self.book, BOOK_OPTIONS["text"])
        fields = {name: getattr(self, name) for name in ("deskew", "center", "per_page", "clean_margin",
                                                          "upscale", "enhance", "flatten")}
        return Options(**{**self.__dict__, **{name: value and name in allowed for name, value in fields.items()}})


def default_output_path(input_path):
    """默认的输出文件：和原文件同目录，文件名后加「（校正版）」。"""
    input_path = Path(input_path)
    return input_path.with_name(input_path.stem + "（校正版）.pdf")


class Cancelled(Exception):
    pass


# ---------------------------------------------------------------- 取得页面图像

def classify_page(doc, page):
    """判断页面的取图方式。返回 (mode, xref)。"""
    images = page.get_images(full=True)
    if not images:
        return "copy", 0            # 没有图像＝不是扫描页，原样保留
    if len(images) == 1 and page.rotation == 0:
        item = images[0]
        xref, smask = item[0], item[1]
        try:
            bbox, m = page.get_image_bbox(item, transform=True)
        except Exception:
            return "render", 0
        covers = abs(bbox & page.rect) >= 0.5 * abs(page.rect)
        upright = m.a > 0 and m.d > 0 and abs(m.b) < 1e-3 and abs(m.c) < 1e-3
        if covers and upright and smask == 0:
            return "native", xref   # 整页就是一张图：直接取原图，避免渲染损失
    return "render", 0


def place_on_page(img, bbox, page_rect, bilevel):
    """把内嵌图按它在页面中的位置贴到一张整页大小的画布上（保持原图分辨率）。

    有的 PDF 扫描图并不占满整页（比如居中放在 A4 页面里）。统一成整页画布后，
    后面的处理就不用区分这两种情况。返回 (画布, 扫描图在画布中的归一化范围, 原图左上角的像素位置)。
    """
    h, w = img.shape[:2]
    tol = 0.005 * max(page_rect.width, page_rect.height)
    if all(abs(a - b) <= tol for a, b in zip(bbox, page_rect)):
        return img, (0.0, 0.0, 1.0, 1.0), (0, 0)
    sx, sy = w / bbox.width, h / bbox.height
    cw, ch = round(page_rect.width * sx), round(page_rect.height * sy)
    ox, oy = round((bbox.x0 - page_rect.x0) * sx), round((bbox.y0 - page_rect.y0) * sy)
    if bilevel:
        paper = 255
    else:
        paper = np.median(img[::8, ::8].reshape(-1, 1 if img.ndim == 2 else 3), axis=0)
    canvas = np.empty((ch, cw) + img.shape[2:], dtype=np.uint8)
    canvas[:] = paper
    x0, y0, x1, y1 = max(ox, 0), max(oy, 0), min(ox + w, cw), min(oy + h, ch)
    canvas[y0:y1, x0:x1] = img[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
    return canvas, (x0 / cw, y0 / ch, x1 / cw, y1 / ch), (ox, oy)


def pixmap_to_array(pix):
    """fitz.Pixmap → numpy 数组（灰度为 HxW，彩色为 HxWx3 的 BGR）。"""
    if pix.colorspace is None or pix.n - pix.alpha > 3:
        pix = fitz.Pixmap(fitz.csRGB, pix)
    if pix.alpha:
        pix = fitz.Pixmap(pix, 0)
    n = pix.n
    buf = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)
    arr = buf[:, : pix.width * n].reshape(pix.height, pix.width, n)
    if n == 1:
        return arr[:, :, 0].copy()
    if n == 2:                       # 灰度以外的双通道极少见，取第一通道
        return arr[:, :, 0].copy()
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def load_native_image(doc, page, xref, bbox):
    """取页面里内嵌的原图。

    有的扫描 PDF 把黑白页存成「图像蒙版」（ImageMask）：没有色彩空间，只记录哪些点要上色，
    上什么颜色、0 和 1 谁代表上色都由页面决定，不能直接当灰度图用（对它做色彩空间转换会报
    "source colorspace must not be None"）。这种图改为按它自身的分辨率把所在区域渲染出来，
    像素和原图一一对应，颜色由 PDF 引擎按页面的设定处理；再二值化去掉渲染的抗锯齿灰边。
    """
    pix = fitz.Pixmap(doc, xref)
    if pix.colorspace is not None:
        return pixmap_to_array(pix)
    matrix = fitz.Matrix(pix.width / bbox.width, pix.height / bbox.height)
    rendered = pixmap_to_array(page.get_pixmap(matrix=matrix, clip=bbox, colorspace=fitz.csGRAY, alpha=False))
    if rendered.shape[:2] != (pix.height, pix.width):       # 取整可能差 1 个像素
        rendered = cv2.resize(rendered, (pix.width, pix.height), interpolation=cv2.INTER_NEAREST)
    return cv2.threshold(rendered, 127, 255, cv2.THRESH_BINARY)[1]


def load_page_image(doc, page, info, dpi):
    if info.mode == "native":
        item = next(it for it in page.get_images(full=True) if it[0] == info.xref)
        bbox = page.get_image_bbox(item)
        img = load_native_image(doc, page, info.xref, bbox)
    else:
        img = pixmap_to_array(page.get_pixmap(dpi=dpi))
    # 三通道但实际是灰度的页面，转成单通道以减小输出体积
    if img.ndim == 3:
        small = img[::8, ::8].astype(np.int16)
        if np.abs(small[:, :, 0] - small[:, :, 1]).max() < 10 and \
           np.abs(small[:, :, 1] - small[:, :, 2]).max() < 10:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if info.mode == "native":
        info.dpi = img.shape[1] / bbox.width * 72
        img, info.scan_rect, info.grid = place_on_page(img, bbox, page.rect, info.bilevel)
        is_jpeg = "DCTDecode" in doc.xref_get_key(info.xref, "Filter")[1]
        info.jpeg_block = (8 if img.ndim == 2 else 16) if is_jpeg else 0
    else:
        info.dpi, info.jpeg_block, info.grid = dpi, 0, (0, 0)
    return img


# ---------------------------------------------------------------- 分析

def to_gray(img):
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def make_ink_mask(gray_small, scan_rect=(0.0, 0.0, 1.0, 1.0)):
    """二值化得到「墨迹」掩码，并清掉不属于版面内容的东西。

    1. 与扫描图边缘相连的成分（扫描黑边、书脊阴影）。注意是扫描图的边缘，
       不是页面的边缘——扫描图可能只占页面的一部分。
    2. 细长的线（纸张边缘的阴影线等）。标题的装饰线也会被去掉，但它本来就在
       文字范围之内，不影响版心的判定。
    3. 远离其他内容的孤立小污点。
    """
    _, ink = cv2.threshold(gray_small, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    h, w = ink.shape

    # 1. 贴边成分（边缘 3 像素以内都算贴边，容许坐标取整的误差）
    x0, y0 = int(np.floor(scan_rect[0] * w)), int(np.floor(scan_rect[1] * h))
    x1, y1 = int(np.ceil(scan_rect[2] * w)), int(np.ceil(scan_rect[3] * h))
    ink[:y0] = 0; ink[y1:] = 0; ink[:, :x0] = 0; ink[:, x1:] = 0
    scan = ink[y0:y1, x0:x1]
    n, labels, _, _ = cv2.connectedComponentsWithStats(scan, connectivity=8)
    m = 3
    border = np.unique(np.concatenate([labels[:m].ravel(), labels[-m:].ravel(),
                                       labels[:, :m].ravel(), labels[:, -m:].ravel()]))
    remove = np.zeros(n, dtype=bool)
    remove[border] = True
    remove[0] = False
    scan[remove[labels]] = 0

    # 2. 细长线：先膨胀把断断续续的线段连起来，再按最小外接矩形的「粗细/长度」判断
    k = 7
    merged = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    erase = np.zeros_like(ink)
    blobs = []
    for c in contours:
        (_, _), (rw, rh), _ = cv2.minAreaRect(c)
        thick, length = min(rw, rh), max(rw, rh)
        if thick <= k + 4 and length >= 0.05 * max(h, w):
            cv2.drawContours(erase, [c], -1, 255, cv2.FILLED)
        else:
            blobs.append(c)

    # 3. 边缘地带的小块：整个落在扫描图边缘 3% 以内的小东西，是黑边的残留或边缘污渍
    zx, zy = 0.03 * (x1 - x0), 0.03 * (y1 - y0)
    small_size = 0.025 * max(h, w)
    inner = []
    for c in blobs:
        bx, by, bw, bh = cv2.boundingRect(c)
        side_zone = bx + bw <= x0 + zx or bx >= x1 - zx     # 左右边缘：正文不可能整个挤在这里
        end_zone = by + bh <= y0 + zy or by >= y1 - zy      # 上下边缘：可能有页眉页码，只去细线和小块
        thin = min(cv2.minAreaRect(c)[1]) <= k + 4
        if side_zone or (end_zone and (thin or max(bw, bh) - k <= small_size)):
            cv2.drawContours(erase, [c], -1, 255, cv2.FILLED)
        else:
            inner.append(c)
    blobs = inner

    # 4. 孤立小污点：自身很小，并且周围一圈之内没有别的内容
    reach = int(0.04 * max(h, w))
    kept = np.zeros_like(ink)
    cv2.drawContours(kept, blobs, -1, 255, cv2.FILLED)
    for c in blobs:
        bx, by, bw, bh = cv2.boundingRect(c)
        if max(bw, bh) > k + 10:
            continue
        around = kept[max(by - reach, 0):by + bh + reach, max(bx - reach, 0):bx + bw + reach]
        own = cv2.countNonZero(kept[by:by + bh, bx:bx + bw])
        if cv2.countNonZero(around) - own == 0:
            cv2.drawContours(erase, [c], -1, 255, cv2.FILLED)
    ink[erase > 0] = 0
    return ink


def rotate(img, angle, border_value=0, flags=cv2.INTER_LINEAR):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=flags,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)


def detect_skew(ink, max_angle):
    """投影轮廓法：文字行完全水平（竖排则列完全垂直）时，投影的起伏最剧烈。

    同时计算行方向和列方向的得分，取峰值更尖锐的一方，因此横排、竖排都适用。
    返回 (角度, 置信度, 是否竖排)。
    """
    def scores(angle):
        r = rotate(ink, angle, flags=cv2.INTER_NEAREST)
        rows = r.sum(axis=1, dtype=np.float64)
        cols = r.sum(axis=0, dtype=np.float64)
        return np.sum(np.diff(rows) ** 2), np.sum(np.diff(cols) ** 2)

    def search(angles):
        s = np.array([scores(a) for a in angles])       # 形状 (N, 2)
        sharp = s.max(axis=0) / (np.median(s, axis=0) + 1e-9)
        axis = int(np.argmax(sharp))
        return angles[int(np.argmax(s[:, axis]))], sharp[axis], axis

    coarse, conf, axis = search(np.arange(-max_angle, max_angle + 1e-6, 0.5))
    fine, _, _ = search(np.arange(coarse - 0.5, coarse + 0.5 + 1e-6, 0.05))
    return float(fine), float(conf), axis == 1


def detect_content_bbox(ink):
    """求版心包围盒（归一化坐标）。返回 (版心, 核心版心, 核心之外的小块)，没有内容时版心为 None。

    先膨胀把文字连成块，再丢掉零星噪点。「版心」包含剩下的全部内容；「核心版心」只算成块的
    内容——膨胀后够长（超过页面长边的 3%）而且够粗（超过 1.2%）的块（文字行、插图）。伸出核心版心之外的小块单独列出来：
    它们可能是页码、目录的序号，也可能是空白处的污渍，单看一页分不清，留给 effective_boxes
    对照全书的标准版心去判断。
    """
    h, w = ink.shape
    merged = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
    pad = 3  # 抵消膨胀造成的外扩

    def box_of(i):
        x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        return (round(float((x + pad) / w), 4), round(float((y + pad) / h), 4),
                round(float((x + stats[i, cv2.CC_STAT_WIDTH] - pad) / w), 4),
                round(float((y + stats[i, cv2.CC_STAT_HEIGHT] - pad) / h), 4))

    def union(boxes):
        if not boxes:
            return None
        a = np.array(boxes)
        return (float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 2].max()), float(a[:, 3].max()))

    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 150]
    # 成块：够长，而且够粗。只长不粗的是线段（纸边阴影的残段等），文字行膨胀后至少有 20 像素高
    big, thick = 0.03 * max(h, w), 0.012 * max(h, w)
    is_core = {i: max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]) > big
               and min(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]) > thick for i in keep}
    full = union([box_of(i) for i in keep])
    core = union([box_of(i) for i in keep if is_core[i]])
    outliers = []
    if core:
        for i in keep:
            b = box_of(i)
            if not is_core[i] and (b[0] < core[0] or b[1] < core[1] or b[2] > core[2] or b[3] > core[3]):
                outliers.append(b)
    return full, core, outliers


DENSE_RATIO = 0.08          # 覆盖某一列的文字行数不到全页行数的这个比例（至少要 3 行），这一列就不算正文


def dense_extent(ink):
    """求正文在水平方向上的范围（归一化），求不出来返回 None。要竖直方向的就传 ink.T。

    做法是数「每一列被多少行文字覆盖」：先按行投影把页面切成一行一行，再看每一行占了哪些列。
    正文的列被几十行覆盖；挂在正文外侧空白里的页码、页眉各占一行，空白处的污渍也只碰到一两行。
    不能比墨迹的多少——有的页码带一块实心的花饰，那几列的墨迹并不比正文少（试过，分不开）。

    只适合沿文字行的方向用：在行堆叠的方向上没有这种「叠了很多行」的性质，段落最后一行可能
    只有两三个字，和页码分不开，硬用会让版心的下边随页忽上忽下。
    """
    h, w = ink.shape
    # 哪些行属于文字行：墨迹要够多。只看「有没有墨迹」不行——页边只要有一道贯穿上下的细线或
    # 一串污点，每一行就都「有墨迹」，整页连成一行
    amount = ink.sum(axis=1) / 255.0
    if not amount.any():
        return None
    rows = np.flatnonzero(amount >= 0.15 * np.median(amount[amount > 0]))
    if not len(rows):
        return None
    # 连续有墨迹的行归为一行文字；中间断 2 个像素以内的不算断（注音、标点会让行投影有细缝）
    breaks = np.flatnonzero(np.diff(rows) > 3)
    runs = [(rows[a], rows[b] + 1) for a, b in zip(np.r_[0, breaks + 1], np.r_[breaks, len(rows) - 1])]
    runs = [(a, b) for a, b in runs if b - a >= 4]                  # 不到 4 个像素高的是噪点，不是一行字
    if len(runs) < 4:                                               # 行数太少，「叠了很多行」无从谈起
        return None
    k = max(5, int(0.015 * w))                                      # 约一个字宽：抹平字与字之间的空隙
    kernel = np.ones(k)
    cover, exact = np.zeros(w), np.zeros(w)
    for a, b in runs:
        occupied = ink[a:b].any(axis=0)
        exact += occupied
        cover += np.convolve(occupied.astype(float), kernel, mode="same") > 0
    # 至少 3 行：页眉和页码常常挂在同一侧（右页的右上角和右下角），那几列会被 2 行覆盖
    dense = np.flatnonzero(cover >= max(3, DENSE_RATIO * len(runs)))
    if not len(dense):
        return None
    # 抹平空隙的同时也把边缘向外抹开了半个字：回到没抹过的覆盖数上，取这一带里第一个和最后一个
    # 「至少被 2 行盖住」的列。不能只看有没有墨迹——页码的花饰可能就贴在正文旁边不到一个字宽的地方
    cols = exact >= 2
    lo_zone = np.flatnonzero(cols[max(dense[0] - k, 0):dense[0] + k + 1])
    hi_zone = np.flatnonzero(cols[max(dense[-1] - k, 0):dense[-1] + k + 1])
    lo = max(dense[0] - k, 0) + lo_zone[0] if len(lo_zone) else dense[0]
    hi = max(dense[-1] - k, 0) + hi_zone[-1] + 1 if len(hi_zone) else dense[-1] + 1
    return float(lo / w), float(hi / w)


def is_full_bleed_color(img, scan_rect=(0.0, 0.0, 1.0, 1.0)):
    """彩色、而且画面一直铺到页边的页（彩色封面、整页彩图）。

    整页图片本来靠「深色内容占满页面 95%」来认，可封面上浅色的渐变底不算深色内容，于是封面被当成
    普通页：标题和插画那一块成了「版心」，拿去和文字页对齐，整页出血的封面被平移了 5%，一边露出
    一条平色带。所以另加一条：页面带颜色（彩度明显的像素占 5% 以上），并且四周 3% 的边缘带里
    像纸的（够亮的）像素不到一半——画面铺到了边上。黑白扫描边不会误判：它是黑的、不带颜色。
    带白边的彩色封面不算，它照常处理，需要的话手动「本页不修正」。只看扫描图的范围。
    """
    if img.ndim != 3:
        return False
    h, w = img.shape[:2]
    x0, y0 = int(scan_rect[0] * w), int(scan_rect[1] * h)
    area = img[y0:int(scan_rect[3] * h):4, x0:int(scan_rect[2] * w):4]
    if min(area.shape[:2]) < 20:
        return False
    chroma = area.max(axis=2).astype(np.int16) - area.min(axis=2).astype(np.int16)
    if (chroma > COLOR_CHROMA).mean() < 0.05:
        return False
    ah, aw = area.shape[:2]
    bx, by = max(int(0.03 * aw), 1), max(int(0.03 * ah), 1)
    lum = to_gray(area)
    bands = np.concatenate([lum[:, :bx].ravel(), lum[:, -bx:].ravel(), lum[:by, :].ravel(), lum[-by:, :].ravel()])
    return bool((bands >= PAPER_LIKE).mean() < 0.5)


def analyze(img, info, max_angle):
    info.full_bleed_color = is_full_bleed_color(img, info.scan_rect)
    gray = to_gray(img)
    scale = ANALYSIS_LONG_SIDE / max(gray.shape)
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) \
        if scale < 1 else gray
    ink = make_ink_mask(small, info.scan_rect)
    if cv2.countNonZero(ink) < 0.001 * ink.size:
        info.mode, info.note = "copy", "空白页或整页图片"
        return
    angle, conf, info.vertical = detect_skew(ink, max_angle)
    if conf < 1.2:
        angle, info.note = 0.0, "倾斜不明确，不旋转"
    info.angle = angle
    ink = rotate(ink, angle, flags=cv2.INTER_NEAREST)
    bbox, core_bbox, outliers = detect_content_bbox(ink)
    if bbox is None:
        info.mode, info.note = "copy", "未检测到版心"
        return
    if bbox[2] - bbox[0] > FULL_BLEED_RATIO and bbox[3] - bbox[1] > FULL_BLEED_RATIO:
        info.mode, info.note = "copy", "内容占满整页"
        return
    info.bbox, info.core_bbox, info.outliers = bbox, core_bbox, outliers
    # 两个方向的密度范围都存下来：哪个方向是文字行的方向，要等全书分析完才知道（单页会误判）
    # 两个方向各求各的，求不出来的那个方向就用版心本身：横排书在竖直方向上本来就求不出来
    # （各列的字上下连成一片，分不出「行」），不能因此把水平方向的结果也丢掉
    xs = dense_extent(ink) or (bbox[0], bbox[2])
    ys = dense_extent(ink.T) or (bbox[1], bbox[3])
    info.dense_bbox = (xs[0], ys[0], xs[1], ys[1])


# ---------------------------------------------------------------- 对齐量的计算

def _reference_of(boxes, infos):
    arr = np.array([b for _, b in boxes])
    ref = {"ext": (np.median(arr[:, 2] - arr[:, 0]), np.median(arr[:, 3] - arr[:, 1])),
           "vertical": sum(p.vertical for p in infos if p.bbox) * 2 > len(boxes)}
    for parity in (0, 1):
        sub = np.array([b for i, b in boxes if i % 2 == parity])
        ref[parity] = np.median(sub if len(sub) >= 4 else arr, axis=0)
    return ref


def _peel(core_lo, core_hi, spans, limit):
    """一个方向上：从核心版心加上全部小块出发，由外向内剥掉小块，直到范围不超过 limit。

    每次剥最外侧的那个；两侧都有时，剥「离里面的内容更远」的一侧——空白处的污渍是孤零零的，
    和内容之间隔着一大段空白；页码、目录的序号紧挨着正文，间隔很小，会留到最后。
    """
    los = sorted((sp for sp in spans if sp[0] < core_lo), key=lambda sp: sp[0])
    his = sorted((sp for sp in spans if sp[1] > core_hi), key=lambda sp: -sp[1])
    while True:
        lo = min([core_lo] + [sp[0] for sp in los])
        hi = max([core_hi] + [sp[1] for sp in his])
        if hi - lo <= limit or not (los or his):
            return lo, hi
        gap_lo = min([core_lo] + [sp[0] for sp in los[1:]]) - los[0][0] if los else -1.0
        gap_hi = his[0][1] - max([core_hi] + [sp[1] for sp in his[1:]]) if his else -1.0
        (los if gap_lo >= gap_hi else his).pop(0)


def effective_boxes(infos, ext):
    """决定每一页实际采用的版心。ext 是全书标准版心的 (宽, 高)。

    空白处的小污渍（一小段竖线、一个墨点）会把版心撑大：框还是居中的，文字却偏到了一边。
    单看一页分不清它是污渍还是页码——大小差不多；但和全书的标准版心一比就清楚了：
    版心明显比标准大，就是被杂质撑大的，把最外侧、最孤立的小块逐个剥掉，剥到大小正常为止。
    带页码的版心本来就和标准一样大，不会被动到。
    """
    result = {}
    for p in infos:
        if not p.bbox:
            continue
        box = list(p.bbox)
        if p.core_bbox and p.outliers:
            for lo, hi, med in ((0, 2, ext[0]), (1, 3, ext[1])):
                if box[hi] - box[lo] > med + OVERSIZE_TOLERANCE:
                    spans = [(b[lo], b[hi]) for b in p.outliers]
                    box[lo], box[hi] = _peel(p.core_bbox[lo], p.core_bbox[hi], spans, med + OVERSIZE_TOLERANCE)
        result[p.index] = tuple(box)
    return result


def compute_reference(infos):
    """统计标准版心：全书版心宽高的中位数，以及奇偶页各自的版心边缘位置中位数。

    每一页有两个版心：
    - ref["content"]：这一页的全部内容，剔掉了空白处的杂质（见 effective_boxes）。涂白页边时用它，
      免得把挂在正文外面的页码也涂掉。
    - ref["boxes"]：对齐用的版心。行堆叠的方向和上面一样；**沿文字行的方向换成按密度求出的正文
      范围**（见 dense_extent）。有的书把页码挂在正文外侧的空白里（左页在左下角、右页在右下角），
      带页码的版心比正文宽一截：拿它居中，正文就会一页偏左一页偏右，没有页码的章首页还会被
      当成「窄了」而贴到一边去。对齐要看的是正文，不是页码。
    标准版心的统计用的是对齐用的版心。
    """
    raw = [(p.index, p.bbox) for p in infos if p.bbox]
    if not raw:
        return None
    first = _reference_of(raw, infos)
    content = effective_boxes(infos, first["ext"])
    lo, hi = (1, 3) if first["vertical"] else (0, 2)        # 文字行的方向：横排书是水平方向
    boxes = {}
    for p in infos:
        if p.index in content:
            box = list(content[p.index])
            if p.dense_bbox:
                box[lo], box[hi] = p.dense_bbox[lo], p.dense_bbox[hi]
            boxes[p.index] = tuple(box)
    # 按密度求出的正文范围只用来**去掉伸到正文外面的东西**（外挂的页码、页眉、污渍），也就是只在
    # 全部内容比标准的正文明显宽时才用（门槛 3%：外挂的页码至少伸出去 5%；而「标准」是按密度求的，
    # 行尾参差不齐的书里它本来就比实际内容窄一点，门槛定在 2% 会误伤正常的页）。其余的页直接用全部内容：整行的文字只有寥寥几行的页
    # （诗歌、对话、字表、目录），整行所在的列达不到「叠了很多行」的门槛，求出来的范围会偏窄
    standard = _reference_of(sorted(boxes.items()), infos)["ext"][0 if lo == 0 else 1]
    for index, whole in content.items():
        if whole[hi] - whole[lo] <= standard + HANGING_TOLERANCE:
            box = list(boxes[index])
            box[lo], box[hi] = whole[lo], whole[hi]
            boxes[index] = tuple(box)
    ref = _reference_of(sorted(boxes.items()), infos)
    ref["boxes"], ref["content"] = boxes, content
    return ref


def page_box(info, ref, key="adjust"):
    """这一页的版心（界面预览里的红框）。

    ref["adjust"]（可选）是用户对个别页的手动微调 {页序: (左, 上, 右, 下 四条边各自的移动量)}，
    加在自动求出的版心上。它只影响这一页自己，不参与全书标准版心的统计——调一页不该带动别的页。
    """
    if not ref:
        return info.bbox
    box = ref["boxes"].get(info.index, info.bbox)
    offsets = ref.get(key, {}).get(info.index)
    if box and offsets:
        box = [min(max(b + o, 0.0), 1.0) for b, o in zip(box, offsets)]
        for lo, hi in ((0, 2), (1, 3)):                 # 两条边不能交叉，至少留 5% 的宽（高）
            if box[hi] - box[lo] < 0.05:
                mid = (box[lo] + box[hi]) / 2
                box[lo], box[hi] = mid - 0.025, mid + 0.025
        box = tuple(box)
    return box


def align_box(info, ref):
    """这一页**对齐用的**版心。调红框本身不移动页面（用户要求：曾经每调一下页面就跟着重新对齐一次）；
    用户点「版心居中」时，那一刻的红框记在 ref["align"] 里（格式同 ref["adjust"]），从此按它对齐。
    之后再调红框，对齐仍按居中那一刻的，直到再点一次。去污、「与邻页相同」看的始终是红框（page_box）。"""
    return page_box(info, ref, "align")


def cleanup_box(info, ref):
    """用户对这一页指定了「去除边缘污染」时，返回它的红框（对齐用的版心），否则返回 None。

    和手动微调一样挂在 ref 上：ref["cleanup"] 是指定了去污的页序集合。
    """
    if ref and info.bbox and info.mode != "copy" and info.index in ref.get("cleanup", ()):
        return page_box(info, ref)
    return None


def box_like_neighbor(info, neighbor, ref):
    """「与前页/后页相同」：算出让这一页的红框和邻页一样所需要的手动微调量 (左, 上, 右, 下)。

    不能直接照搬邻页红框的坐标：左右页的版心在扫描图上本来就差几个百分点，每一页的扫描位置
    还有 1% 多的抖动。所以分三步：
    1. 取邻页的红框（含它自己的手动微调）——要的是它的**大小**；
    2. 邻页和本页可能分属左右页，按全书统计的左右页位置差平移过来；
    3. 平移后如果有一条边和本页自动检测到的边很接近（差 3% 以内），整体挪过去贴齐那条边，
       把扫描位置的抖动消掉。两条边都对不上就保持第 2 步的结果，留给用户用箭头微调。
    """
    target = list(page_box(neighbor, ref))
    auto = ref["boxes"][info.index]
    a, b = ref[0], ref[1]
    for lo, hi in ((0, 2), (1, 3)):
        gap = ((a[lo] - b[lo]) + (a[hi] - b[hi])) / 2          # 左右页的位置差（两条边取平均，不改变大小）
        # 页序的奇偶不可靠（前置部分多一页少一页就反了），所以不押注谁是左页谁是右页：
        # 「同一侧」「差一个位置差」「反方向差一个位置差」三种都试，取和本页检测到的边最吻合的
        best = None
        for move in (0.0, gap, -gap):
            d_lo, d_hi = auto[lo] - (target[lo] + move), auto[hi] - (target[hi] + move)
            snap = d_lo if abs(d_lo) <= abs(d_hi) else d_hi
            if best is None or abs(snap) < abs(best[1]) - 1e-9:
                best = (move, snap)
        move, snap = best
        same_size = abs((target[hi] - target[lo]) - (auto[hi] - auto[lo])) <= FULL_PAGE_TOLERANCE
        if same_size:
            # 本页检测到的红框和邻页的一样大：那它的位置是可信的，直接对准（两条边的差取平均）。
            # 扫描位置的上下抖动可以有好几个百分点，这时不能拿 3% 的贴边容差去卡它
            move = ((auto[lo] - target[lo]) + (auto[hi] - target[hi])) / 2
        elif abs(snap) <= EDGE_MATCH_TOLERANCE:
            move += snap
        target[lo] += move
        target[hi] += move
    return tuple(round(float(t - a), 5) for t, a in zip(target, auto))


def content_box(info, ref):
    """这一页全部内容的范围（含挂在正文外面的页码，不含空白处的杂质）。"""
    return ref["content"].get(info.index, info.bbox) if ref else info.bbox


def axis_shift(lo, hi, med_ext, edges, along_lines):
    """单个方向上的平移量（归一化）。

    edges 是 [(本页所属奇偶类的版心两边位置中位数), (另一类的)]；
    along_lines 表示这个方向是不是沿着文字行的方向（横排书的水平方向）。

    1. 整页（版心和标准版心一样大）：直接居中。
    2. 沿文字行的方向上只是略窄（目录、诗歌、整体缩进的段落）：看它窄在哪一边。通常是一边和
       正文对齐、缩进全在另一边——那就把对齐的那一边贴齐标准版心，缩进原样保留，不能居中
       （居中会把缩进平摊到两边，这一页就和前后页错开了）。两边缩得差不多的才是居中排版的
       内容，直接居中。扫描位置的抖动一般只有 1% 多，比缩进量小得多，所以分得清。
    3. 半页（章末、章首、没有页眉的页等）：直接居中会让文字偏离它该在的位置。如果它
       有一边本来就和标准版心的边缘基本重合（章末页的上边、章首页的下边），就把这一边
       贴齐标准版心。先和本页所属的奇偶类比，对不上再和另一类比——页序的奇偶不一定
       可靠：前置部分和正文之间多一页或少一页，左右页的对应关系就反过来了。
    4. 和两类都对不上（前言、落款等版式特殊的页），无从判断它的版心在哪里：只是略小的
       就直接居中；小很多的只按全书的平均偏移量移动（取两类的平均，不押注它属于哪一类）。
    """
    ext = hi - lo
    centered = (1 - ext) / 2 - lo
    if ext >= med_ext - FULL_PAGE_TOLERANCE:
        return centered
    slightly_smaller = ext >= med_ext - NARROW_TOLERANCE
    std_lo = (1 - med_ext) / 2
    std_hi = std_lo + med_ext
    if along_lines and slightly_smaller:
        # 取两边里更吻合的那一类来比（页序的奇偶不可靠）
        d_lo, d_hi = min(((abs(lo - e[0]), abs(hi - e[1])) for e in edges), key=min)
        if min(d_lo, d_hi) <= FLUSH_RATIO * max(d_lo, d_hi):
            return std_lo - lo if d_lo <= d_hi else std_hi - hi
        return centered
    for med_lo, med_hi in edges:
        d_lo, d_hi = abs(lo - med_lo), abs(hi - med_hi)
        if min(d_lo, d_hi) <= EDGE_MATCH_TOLERANCE:
            return std_lo - lo if d_lo <= d_hi else std_hi - hi
    if slightly_smaller:
        return centered
    return std_lo - sum(e[0] for e in edges) / len(edges)


def compute_shift(info, ref, per_page):
    x0, y0, x1, y1 = align_box(info, ref)
    if per_page or ref is None:
        return (1 - (x1 - x0)) / 2 - x0, (1 - (y1 - y0)) / 2 - y0
    own, other = ref[info.index % 2], ref[1 - info.index % 2]
    vertical = ref["vertical"]      # 按全书多数页的排版方向，不看单页（单页可能误判）
    return (axis_shift(x0, x1, ref["ext"][0], [(own[0], own[2]), (other[0], other[2])], not vertical),
            axis_shift(y0, y1, ref["ext"][1], [(own[1], own[3]), (other[1], other[3])], vertical))


# ---------------------------------------------------------------- 输出

def paper_tone(img):
    """去污用的纸面估计：每一处「没有墨迹的纸面」长什么样（和 img 同样大小、同样的通道数）。

    做一次大核的闭运算（先膨胀后腐蚀），比核小的深色东西——文字、污点、黑边、阴影——都会被
    周围的纸色填平；平缓的明暗变化则原样保留。纸面的变化很平缓，缩小 4 倍算，快 16 倍。
    核的半径要明显大于黑边的宽度，最靠边的像素才够得着里面的纸色：12% 的核能填平宽度在页面 5% 以内、
    笔直贴着页边的黑边（6% 的核对 3% 宽的黑边刚好差一两列像素，没旋转的页上露过馅）。
    **不能用来找平**：缩小之后网点糊成一片浅灰，闭运算看不到点与点之间的白，整片天空会被当成
    发灰的纸面拉亮，网点拉成格子（用户报的）。找平用 paper_map。
    """
    h, w = img.shape[:2]
    f = 4
    small = cv2.resize(img, (max(w // f, 1), max(h // f, 1)), interpolation=cv2.INTER_AREA)
    k = max(int(0.12 * min(small.shape[:2])) | 1, 9)
    paper = cv2.morphologyEx(small, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return cv2.resize(cv2.GaussianBlur(paper, (k, k), 0), (w, h), interpolation=cv2.INTER_LINEAR)


def paper_map(gray, cells=(64, 48)):
    """找平用的纸面估计（灰度，和 gray 同样大小）：把页面分成 cells = (行数, 列数) 个小格，每格取
    最亮的 5% 像素的亮度当这一格的纸色。

    在原分辨率上做、不缩小：网点之间的白缝就是纸，只有在原分辨率上才看得见（缩小后点和缝混成一片
    浅灰）。文字、排线、网点占不满一格，最亮的 5% 总是纸；整格实心黑（估出来不够亮的格子）不算，
    由周围的格子插值。格子要比订口阴影窄得多（阴影占页宽 5%～10%），48 列每列约 2%：一格里阴影的
    变化不到几级，最亮的 5% 才代表得了这一格；格子再大，页边那一格估出来的是它靠里的一侧，页边就找不平。
    最后略作平滑再放大回原大小，纸色是平缓变化的，不会出现格子的边。
    """
    h, w = gray.shape
    rows, cols = cells
    ch, cw = -(-h // rows), -(-w // cols)
    padded = cv2.copyMakeBorder(gray, 0, rows * ch - h, 0, cols * cw - w, cv2.BORDER_REFLECT)
    blocks = padded.reshape(rows, ch, cols, cw).transpose(0, 2, 1, 3).reshape(rows, cols, -1)
    paper = np.percentile(blocks, 95, axis=2).astype(np.float32)
    valid = (paper >= PAPER_MIN).astype(np.float32)
    if valid.sum() == 0:
        return np.full((h, w), float(np.median(paper)), np.float32)
    # 实心黑的格子用周围有纸的格子填（归一化卷积：先只对有效格子求加权平均）
    k = 0
    filled = paper * valid
    weight = valid.copy()
    while (weight == 0).any() and k < 6:
        k += 1
        filled = cv2.GaussianBlur(paper * valid, (2 * k + 1,) * 2, 0, borderType=cv2.BORDER_REPLICATE)
        weight = cv2.GaussianBlur(valid, (2 * k + 1,) * 2, 0, borderType=cv2.BORDER_REPLICATE)
        filled = np.where(weight > 0, filled / np.maximum(weight, 1e-6), 0)
        paper, valid = np.where(valid > 0, paper, filled), np.maximum(valid, (weight > 0).astype(np.float32))
    paper = np.where(valid > 0, paper, float(np.median(paper[valid > 0])))
    paper = cv2.GaussianBlur(paper, (3, 3), 0, borderType=cv2.BORDER_REPLICATE)
    return cv2.resize(paper, (w, h), interpolation=cv2.INTER_CUBIC)


PAPER_MIN = 160             # 估出的纸色至少这么亮才算纸：更暗的是比闭运算的核还大的实心黑块，不是纸
FLATTEN_MIN_SHADE = 12      # 页边的纸色比中部暗这么多级以上，「纸面找平」才有明显的改善


def measure_shade(gray, scan_rect=(0.0, 0.0, 1.0, 1.0)):
    """灰度/彩色页：页边的纸色比中部暗多少级（0 = 均匀）。在分析用的缩略图上量，只看扫描图的范围。

    漫画扫描件常有一道订口阴影（页宽的 5%～10%，二三十级），每页位置固定，看着灰蒙蒙的；
    「纸面找平」就是冲它来的，所以分析时先量出来，找平只给有明显灰影的页，建议值也据此而定。
    四条边各取一条 8% 宽的带子，和中部比；带子里估出的纸色不够亮的（整格实心黑）不算。
    """
    h, w = gray.shape
    x0, y0 = int(scan_rect[0] * w), int(scan_rect[1] * h)
    area = gray[y0:int(scan_rect[3] * h), x0:int(scan_rect[2] * w)]
    if min(area.shape) < 40:
        return 0.0
    paper = paper_map(area)
    ah, aw = paper.shape
    center = paper[int(0.3 * ah):int(0.7 * ah), int(0.3 * aw):int(0.7 * aw)]
    center = center[center >= PAPER_MIN]
    if center.size < 100:
        return 0.0
    mid = float(np.median(center))
    bx, by = max(int(0.08 * aw), 1), max(int(0.08 * ah), 1)
    darkest = mid
    for band in (paper[:, :bx], paper[:, -bx:], paper[:by, :], paper[-by:, :]):
        lit = band[band >= PAPER_MIN]
        if lit.size >= 0.2 * band.size:
            darkest = min(darkest, float(np.median(lit)))
    return max(mid - darkest, 0.0)


def flatten_paper(img):
    """「纸面找平」：每一处按「当地的纸色 → 白」拉亮。页边的灰影没了，页与页的观感一致，
    墨线、网点相对纸面的深浅不变（是乘一个系数，不是加减）。不做二值化、不锐化——漫画的网点经不起。
    纸色用 paper_map 估：实心黑的地方按周围的纸色算，乘上去仍然是黑，中等的灰和周围一样略微拉亮。"""
    paper = np.maximum(paper_map(to_gray(img)), 128.0)
    gain = 255.0 / paper
    if img.ndim == 3:
        gain = gain[..., None]
    return np.clip(img.astype(np.float32) * gain + 0.5, 0, 255).astype(np.uint8)


def flatten_of(info, opts, skip=False):
    """这一页要不要做纸面找平：选项开着、是灰度/彩色页、用户没有指定「本页不修正」。

    不按 info.shade 逐页取舍：找平不像显示增强那样有体积的代价，而漫画要的是页与页一致——
    只找平有灰影的页，翻到没灰影的页纸色就从白跳回灰。灰影的多少只用来给建议（recommend_flatten）。
    """
    return bool(opts.flatten and not info.bilevel and not skip and info.mode != "copy")


def remove_margin_stains(img, rect, bilevel):
    """把 rect (x0, y0, x1, y1，像素) 以外疑似墨迹的地方，按周围干净纸面的颜色重新画上。就地修改 img。

    黑白二值页的纸面就是白色，框外直接涂白。灰度/彩色页先估计「没有墨迹的纸面长什么样」：
    做一次大核的闭运算（先膨胀后腐蚀），比核小的深色东西——污点、黑边、阴影——都会被周围的
    纸色填平；再和原图比，明显比纸面深的像素就是疑似墨迹，换成那里的纸色。这样填上去的是
    **当地的**纸色（扫描件的纸面往往一边亮一边暗），不是全页统一的一个颜色，也不动干净的地方。
    """
    h, w = img.shape[:2]
    x0, y0, x1, y1 = rect
    outside = np.ones((h, w), dtype=bool)
    outside[y0:y1, x0:x1] = False
    if bilevel:
        img[outside] = 255
        return img
    paper = paper_tone(img)
    stain = (to_gray(paper).astype(np.int16) - to_gray(img).astype(np.int16)) > 24
    stain = cv2.dilate((stain & outside).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & outside
    img[stain] = paper[stain]
    return img


def transform_image(img, info, angle, shift, clean_margin, scale=1, bbox=None, cleanup=None, keep_scale=False,
                    nearest=False):
    """对整页图像做「旋转 + 平移」。scale > 1 时同时放大输出（见 Options.upscale）。

    不需要旋转的页只做整像素平移、不插值，像素原样搬运，完全无损。

    原图是 JPEG 时，平移量还要凑成编码块（8 或 16 像素）的整数倍，让原图的块网格和输出的块
    网格重合：这样重新压缩几乎不再损失（实测 PSNR 从 34 dB 提高到 42 dB），文件也小三成。
    代价是居中的精度从 1 像素变成半个块，约占页宽的 0.3%，看不出来。
    """
    h, w = img.shape[:2]
    dx, dy = shift[0] * w, shift[1] * h
    bg = np.median(img[::8, ::8].reshape(-1, 1 if img.ndim == 2 else 3), axis=0)
    bg = tuple(float(v) for v in bg)
    if angle == 0.0:
        # 不旋转就不用放大（放大是为了减轻旋转带来的毛刺）；keep_scale 是「显示增强」要的放大，照做
        scale, interp = (scale, cv2.INTER_CUBIC) if keep_scale and scale > 1 else (1, cv2.INTER_NEAREST)
        dx, dy = round(dx), round(dy)
        if info.jpeg_block:
            b, (gx, gy) = info.jpeg_block, info.grid
            dx = round((dx + gx) / b) * b - gx
            dy = round((dy + gy) / b) * b - gy
    else:
        interp = cv2.INTER_CUBIC
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    m[0, 2] += (scale - 1) * w / 2 + scale * dx
    m[1, 2] += (scale - 1) * h / 2 + scale * dy
    w, h = w * scale, h * scale
    if nearest:                                 # 最近邻：每个像素原样变大，不产生新的灰度（保护网点图案用）
        interp = cv2.INTER_NEAREST
    out = cv2.warpAffine(img, m, (w, h), flags=interp,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=bg)
    if clean_margin:
        pad = 0.02
        bbox = bbox or info.bbox            # 传进来的是全部内容的范围（content_box）：含外挂的页码，不含杂质
        x0 = max(int((bbox[0] + shift[0] - pad) * w), 0)
        y0 = max(int((bbox[1] + shift[1] - pad) * h), 0)
        x1 = min(int((bbox[2] + shift[0] + pad) * w), w)
        y1 = min(int((bbox[3] + shift[1] + pad) * h), h)
        mask = np.ones((h, w), dtype=bool)
        mask[y0:y1, x0:x1] = False
        out[mask] = bg if img.ndim == 3 else bg[0]
    if cleanup:                                 # cleanup 是这一页的红框（cleanup_box），去掉它外面的污染
        pad = 0.004
        rect = (max(int((cleanup[0] + shift[0] - pad) * w), 0), max(int((cleanup[1] + shift[1] - pad) * h), 0),
                min(int((cleanup[2] + shift[0] + pad) * w), w), min(int((cleanup[3] + shift[1] + pad) * h), h))
        out = remove_margin_stains(out, rect, info.bilevel)
    return out


# ---------------------------------------------------------------- 显示增强

SMOOTH_BELOW_DPI = 240      # 分辨率低于此的黑白页值得放大平滑：正常阅读的缩放比例下就能看到锯齿
NOISE_MIN = 120             # 噪点 + 小孔至少这么多，并且不少于字数的 8%，才值得清理
HALFTONE_MAX = 0.10         # 网点图案占到页面的 10% 以上：这一页以图为主，整页不增强（小块的图案只是局部让开）


def _speck_limit(dpi):
    """多大的成分算噪点（面积，像素）。180 DPI 下 2 个像素，随分辨率的平方放大；
    定得很保守：标点、浊点、注音假名的笔画都比它大。"""
    return max(2, round(2 * (dpi / 180) ** 2))


def pattern_regions(ink, dpi, scale=1):
    """网点图案的区域（布尔掩码）：抖动出来的页码花饰、插图。特征是噪点大小的小点、小孔密集成片。
    这些地方做什么增强都是帮倒忙：孔被填上成了死黑，点被当成噪点抹掉，平滑把网点糊成一团。"""
    limit = 4 * _speck_limit(dpi) * scale * scale
    tiny = np.zeros(ink.shape, np.float32)
    for binary, connectivity in ((ink, 8), (1 - ink, 4)):
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=connectivity)
        small = stats[:, cv2.CC_STAT_AREA] <= limit
        small[0] = False
        tiny[small[labels]] = 1
    window = (max(round(dpi / 6) * scale, 15) | 1,) * 2
    dense = cv2.boxFilter(tiny, -1, window) > 0.03
    return cv2.dilate(dense.astype(np.uint8), np.ones(window, np.uint8)).astype(bool)


def find_noise(bw, dpi, scale=1):
    """找出可以放心清掉的杂质，返回 (噪点的掩码, 小孔的掩码)。bw 是二值图，scale 是它相对原图放大的倍数。

    光看大小不行：浊点、标点、字母 i 上的点和灰尘一样小；注音假名「あ」「ぬ」里的小圈和笔画上的
    缺损一样小。清错了字就更难认。所以只认两种十拿九稳的：
    - 噪点：够小，而且**孤立**——周围一圈之内没有任何别的墨迹。紧挨着字的小点一律不动。
    - 小孔：够小，而且**四周被厚厚的墨迹包着**——粗笔画里面的缺损。被细笔画围起来的空白是字形
      本身，不动。
    另外，小点、小孔**密集成片**的地方是网点图案（抖动出来的页码花饰、插图），整块都不动：
    那些孔是图案的一部分，填掉就成了一块死黑。实测真实的扫描书里绝大多数「小孔」都属于这种。
    """
    ink = (bw < 128).astype(np.uint8)
    limit = _speck_limit(dpi) * scale * scale
    reach = max(4, round(dpi / 40)) * scale                     # 噪点周围这么远之内不能有别的墨迹
    wall = 2 * scale                                            # 小孔四周的墨迹至少这么厚

    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    small = stats[:, cv2.CC_STAT_AREA] <= limit
    small[0] = False
    specks = small[labels]
    n_w, white_labels, white_stats, _ = cv2.connectedComponentsWithStats(1 - ink, connectivity=4)
    small_w = white_stats[:, cv2.CC_STAT_AREA] <= limit
    small_w[0] = False
    holes = small_w[white_labels]

    pattern = pattern_regions(ink, dpi, scale)                  # 网点图案整块保护起来
    specks &= ~np.isin(labels, np.unique(labels[specks & pattern]))
    holes &= ~np.isin(white_labels, np.unique(white_labels[holes & pattern]))

    others = (ink.astype(bool) & ~specks).astype(np.uint8)
    near = cv2.dilate(others, np.ones((2 * reach + 1,) * 2, np.uint8)).astype(bool)
    crowded = np.unique(labels[specks & near])                  # 旁边有字的小点：可能是笔画的一部分
    specks &= ~np.isin(labels, crowded)

    labels = white_labels
    ring = cv2.dilate(holes.astype(np.uint8), np.ones((2 * wall + 1,) * 2, np.uint8)).astype(bool) & ~holes
    thin = cv2.dilate((ring & ~ink.astype(bool)).astype(np.uint8), np.ones((2 * wall + 1,) * 2, np.uint8)).astype(bool)
    holes &= ~np.isin(labels, np.unique(labels[holes & thin]))  # 包着它的墨迹不够厚：是字形里的空白
    return specks, holes


def assess_enhancement(gray, dpi):
    """判断一张黑白二值页做「显示增强」有没有明显的改善、适合用哪种方法。

    返回方法的组合（用 + 连接），空字符串表示不值得增强：
    - "smooth2" / "smooth3"：放大 2 / 3 倍、平滑、重新二值化。给分辨率低的页：台阶状的笔画边缘变平滑。
      分辨率够的页不做——正常阅读时看不出区别，文件却要大几倍。
    - "clean"：去掉孤立的噪点、填上粗笔画里的小孔（见 find_noise）。给杂质多的页。**不做「接断笔」**：
      汉字笔画密，分辨率低的时候「接上断笔」和「把两笔粘在一起」只差一个像素。
    - "vector"：把轮廓转成平滑的曲线，任意放大都没有锯齿。只给以大字、线条为主的页（扉页、标题页）：
      小字的点阵轮廓本来就不准，矢量化之后有「融化」的感觉，曲线的数据量也太大。
    带网点图（抖动出来的照片、花饰）的页一律不增强：网点会被当成噪点抹掉、被平滑糊成一团。
    """
    ink = (gray < 128).astype(np.uint8)
    h, w = ink.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if n < 2:
        return ""
    area, height = stats[1:, cv2.CC_STAT_AREA], stats[1:, cv2.CC_STAT_HEIGHT]
    limit = _speck_limit(dpi)

    if float(pattern_regions(ink, dpi).mean()) > HALFTONE_MAX:
        return ""

    glyphs = int(((height >= 6) & (height <= 0.2 * h) & (area >= 12)).sum())
    specks, holes = find_noise(gray, dpi)                       # 只数真的会被清掉的那些
    noise = (cv2.connectedComponents(specks.astype(np.uint8), connectivity=8)[0] - 1
             + cv2.connectedComponents(holes.astype(np.uint8), connectivity=4)[0] - 1)
    big = height > 0.04 * h
    components = int((area > limit).sum())
    if components and area[big].sum() >= 0.6 * area.sum() and components < 600:
        return "vector"
    methods = []
    if glyphs >= 50 and dpi < SMOOTH_BELOW_DPI:
        methods.append("smooth3" if dpi < 200 else "smooth2")
    if noise >= NOISE_MIN and noise >= 0.08 * glyphs:
        methods.append("clean")
    return "+".join(methods)


def enhancement_of(info, opts, skip=False):
    """这一页实际要做的增强：选项开着、这一页值得增强、用户没有指定「本页不修正」。"""
    return info.enhance if opts.enhance and info.bilevel and not skip and info.mode != "copy" else ""


def enhancement_scale(method):
    return 3 if "smooth3" in method else 2 if ("smooth2" in method or "vector" in method) else 1


ENHANCE_NAMES = {"smooth2": "平滑放大 ×2", "smooth3": "平滑放大 ×3", "clean": "去噪点", "vector": "矢量化"}


def describe_enhancement(method):
    return "、".join(ENHANCE_NAMES[m] for m in method.split("+")) if method else ""


def enhance_bitmap(img, method, scale, dpi, plain=None):
    """对已经变换（放大）好的黑白页做增强，返回二值图。plain 是同一页用最近邻放大的版本。"""
    gray = to_gray(img)
    bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
    if scale > 1:
        # 放大之后轻轻糊一下再划黑白：台阶变成平滑的曲线。网点图案的区域让开，用最近邻放大的版本——
        # 三次插值会把规则的网点拉成斜纹
        smooth = cv2.threshold(cv2.GaussianBlur(gray, (0, 0), 0.45 * scale), 127, 255, cv2.THRESH_BINARY)[1]
        plain = bw if plain is None else cv2.threshold(to_gray(plain), 127, 255, cv2.THRESH_BINARY)[1]
        pattern = pattern_regions((plain < 128).astype(np.uint8), dpi, scale)
        bw = np.where(pattern, plain, smooth)
    if "clean" in method:
        specks, holes = find_noise(bw, dpi, scale)
        bw[specks] = 255
        bw[holes] = 0
    return bw


def vector_stream(bw, width, height):
    """把二值图里的墨迹轮廓转成 PDF 的路径（内容流）。width/height 是页面的尺寸（点）。

    轮廓先略作简化，再用 Catmull-Rom 样条换算成三次贝塞尔曲线；拐角（转角小于 120°）处不加
    切线，保持尖角。所有轮廓（外轮廓和里面的孔）放进同一个路径，用奇偶规则填充，孔自然是空的。
    """
    contours, _ = cv2.findContours(255 - bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    sx, sy = width / bw.shape[1], height / bw.shape[0]
    parts = ["0 g"]
    for contour in contours:
        pts = cv2.approxPolyDP(contour, 0.6, True).reshape(-1, 2).astype(float)
        if len(pts) < 3:
            continue
        pts = np.column_stack([pts[:, 0] * sx, height - pts[:, 1] * sy])
        prev, nxt = np.roll(pts, 1, axis=0), np.roll(pts, -1, axis=0)
        a, b = pts - prev, nxt - pts
        cos = (a * b).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
        tangent = (nxt - prev) / 6
        tangent[cos < 0.5] = 0                                   # 转角小于 120° 的是真的拐角，不圆滑
        c1, c2 = pts + tangent, nxt - np.roll(tangent, -1, axis=0)
        parts.append(f"{pts[0, 0]:.2f} {pts[0, 1]:.2f} m")
        parts.extend(f"{c1[i, 0]:.2f} {c1[i, 1]:.2f} {c2[i, 0]:.2f} {c2[i, 1]:.2f} {nxt[i, 0]:.2f} {nxt[i, 1]:.2f} c"
                     for i in range(len(pts)))
        parts.append("h")
    parts.append("f*")
    return "\n".join(parts).encode("ascii")


def render_page(doc, info, ref, opts, angle, shift, skip=False):
    """取图、变换、增强，得到这一页最终的图像。返回 (图像, 实际用的增强方法)。"""
    method = enhancement_of(info, opts, skip)
    grow = enhancement_scale(method)                             # 增强本身要的放大，旋转不旋转都照做
    flat = flatten_of(info, opts, skip)
    scale = max(grow, 2 if opts.upscale and info.bilevel else 1)
    if angle == 0.0:
        # 不旋转的页 transform_image 不理会「2 倍分辨率」，出来的图只放大了 grow 倍。这里要跟着改，
        # 否则只去噪点的页会把原尺寸的图当成放大过的：白白糊一遍，噪点的尺寸上限也大了 4 倍
        scale = grow
    source = load_page_image(doc, doc[info.index], info, opts.dpi)
    if flat:                                        # 先找平再变换：旋转补的边就是白的
        source = flatten_paper(source)
    # 不修正的页不受全局的「版心外涂成纸色」影响；它只接受用户逐页指定的去污
    kwargs = dict(scale=scale, bbox=content_box(info, ref), cleanup=cleanup_box(info, ref), keep_scale=grow > 1)
    img = transform_image(source, info, angle, shift, opts.clean_margin and not skip, **kwargs)
    if method:
        plain = None
        if scale > 1:
            plain = transform_image(source, info, angle, shift, opts.clean_margin and not skip, nearest=True, **kwargs)
        img = enhance_bitmap(img, method, scale, info.dpi, plain)
    return img, method


def add_output_page(out, rect, img, info, opts, method):
    """把一页写进输出的 PDF：矢量化的页写成路径，其余的写成图像。"""
    page = out.new_page(width=rect.width, height=rect.height)
    if "vector" in method:
        xref = out.get_new_xref()
        out.update_object(xref, "<<>>")
        out.update_stream(xref, vector_stream(img, rect.width, rect.height))
        page.set_contents(xref)
    else:
        page.insert_image(page.rect, keep_proportion=False, stream=encode_image(img, info.bilevel, opts.quality))
    return page


def enhancement_summary(infos):
    """分析完之后的一句话：全书有多少页值得增强、各用什么方法。"""
    bilevel = [p for p in infos if p.bbox and p.bilevel]
    if not bilevel:
        return "「显示增强」只对黑白二值页有效，本书没有这样的页。"
    chosen = [p.enhance for p in bilevel if p.enhance]
    if not chosen:
        return f"「显示增强」：{len(bilevel)} 页黑白页都已经足够清楚（或带网点图），增强不会有明显的改善。"
    counts = {}
    for method in chosen:
        for m in method.split("+"):
            counts[ENHANCE_NAMES[m]] = counts.get(ENHANCE_NAMES[m], 0) + 1
    detail = "，".join(f"{name} {n} 页" for name, n in counts.items())
    return f"「显示增强」：{len(bilevel)} 页黑白页里有 {len(chosen)} 页可以明显改善（{detail}）。"


def flatten_summary(infos):
    """分析完之后关于「纸面找平」的一句话（漫画）。"""
    value, reason = recommend_flatten(infos)
    if value is None:
        return reason + "。"
    return ("「纸面找平」建议开启：" if value else "「纸面找平」不需要：") + reason + "。"


def encode_image(img, bilevel, quality):
    if bilevel:
        # 原图是黑白二值：重新二值化后存成 1bit PNG，保持锐利且体积小
        _, bw = cv2.threshold(to_gray(img), 127, 255, cv2.THRESH_BINARY)
        ok, buf = cv2.imencode(".png", bw, [cv2.IMWRITE_PNG_BILEVEL, 1])
    else:
        # OPTIMIZE: 按图像内容生成哈夫曼表，画质不变、体积小 5～10%
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality, cv2.IMWRITE_JPEG_OPTIMIZE, 1])
    if not ok:
        raise RuntimeError("图像编码失败")
    return buf.tobytes()


# ---------------------------------------------------------------- 主流程

def analyze_page(doc, index, opts):
    """第 1 遍的单页处理：判定取图方式，检测倾斜角度和版心。"""
    page = doc[index]
    info = PageInfo(index=index)
    info.mode, info.xref = classify_page(doc, page)
    if info.mode == "copy":
        info.note = "非扫描页"
        return info
    if info.mode == "native":
        info.bilevel = doc.extract_image(info.xref).get("bpc") == 1
    img = load_page_image(doc, page, info, opts.dpi)
    analyze(img, info, opts.max_angle)
    if info.bbox and info.bilevel:
        info.enhance = assess_enhancement(to_gray(img), info.dpi)
    elif info.bbox:
        gray = to_gray(img)
        scale = ANALYSIS_LONG_SIDE / max(gray.shape)
        small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
        info.shade = round(measure_shade(small, info.scan_rect), 1)
    return info


def analyze_document(doc, indices, opts, progress=None, cancelled=None):
    infos = []
    for k, i in enumerate(indices, 1):
        if cancelled and cancelled():
            raise Cancelled
        try:
            infos.append(analyze_page(doc, i, opts))
        except Exception as e:              # 一页读不了不该让整本书失败：这一页原样保留，其余照常
            infos.append(PageInfo(index=i, mode="copy", note=f"无法读取，原样保留（{type(e).__name__}: {e}）"))
        if progress:
            progress(k, len(indices))
    return infos


ANGLE_STEPS = (0.1, 0.2, 0.3, 0.5, 1.0)


# JPEG 标准亮度量化表（IJG / libjpeg，质量 50）。OpenCV 编码用的就是它按质量缩放后的表
_STD_LUMA = np.array([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], dtype=float)
_ZIGZAG = [0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6,
           7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31,
           39, 46, 53, 60, 61, 54, 47, 55, 62, 63]


def jpeg_quality_of(data):
    """从 JPEG 数据的亮度量化表反推它大致相当于 libjpeg 的哪一档质量（1~100）。读不出来返回 None。

    JPEG 文件里并不记录"质量"这个数，只有量化表；拿它和标准表比出缩放倍数，再按 libjpeg 的
    公式换算回质量。扫描仪、Adobe 等用的不是标准表，得到的只是一个相当值，做建议够用了。
    """
    i = 2
    while i + 4 <= len(data) and data[i] == 0xFF:
        marker = data[i + 1]
        if marker == 0xDA:                      # 图像数据开始，后面不会再有量化表
            break
        length = struct.unpack(">H", data[i + 2:i + 4])[0]
        if marker == 0xDB:
            j = i + 4
            while j < i + 2 + length:
                precision, table_id = data[j] >> 4, data[j] & 15
                n = 128 if precision else 64
                if table_id == 0:
                    values = np.frombuffer(data[j + 1:j + 1 + n], dtype=">u2" if precision else np.uint8)
                    table = np.zeros(64)
                    table[_ZIGZAG] = values
                    scale = float(np.mean(100.0 * table / _STD_LUMA))
                    return (200 - scale) / 2 if scale <= 100 else 5000 / scale
                j += 1 + n
        i += 2 + length
    return None


def source_profile(doc, infos, samples=15):
    """抽查原 PDF 里图像的压缩方式：是不是 JPEG、质量大概多少。只读各图的文件头，很快。"""
    pages = [p for p in infos if p.bbox and p.mode == "native"]
    step = max(len(pages) / samples, 1)
    picked = [pages[int(k * step)] for k in range(min(samples, len(pages)))]
    qualities = []
    for info in picked:
        if "DCTDecode" in doc.xref_get_key(info.xref, "Filter")[1]:
            q = jpeg_quality_of(doc.xref_stream_raw(info.xref))
            if q:
                qualities.append(q)
    jpeg = bool(picked) and len(qualities) * 2 > len(picked)

    # 需要渲染的页（一页里有多张图、带旋转等，取不了内嵌原图）：量一下这些页里图像的实际分辨率
    rendered = [p for p in infos if p.mode == "render"]
    step = max(len(rendered) / samples, 1)
    dpis = []
    for info in [rendered[int(k * step)] for k in range(min(samples, len(rendered)))]:
        page = doc[info.index]
        page_dpis = []
        for item in page.get_images(full=True):
            try:
                width = page.get_image_bbox(item).width
            except Exception:
                continue
            if width > 1:
                page_dpis.append(item[2] / width * 72)
        if page_dpis:
            dpis.append(max(page_dpis))
    return {"jpeg": jpeg, "quality": float(np.median(qualities)) if jpeg else None,
            "render_pages": len(rendered), "render_dpi": float(np.median(dpis)) if dpis else None}


def _mostly_bilevel(infos):
    pages = [p for p in infos if p.bbox]
    bilevel = [p for p in pages if p.bilevel]
    return len(bilevel) * 2 > len(pages), bilevel


def recommend_min_angle(infos, profile=None):
    """根据全书的分析结果，给「小于此角度不旋转」提一个建议值。

    旋转需要重新采样。黑白二值图每个像素非黑即白，旋转后笔画边缘会变糙，分辨率越低越明显；
    JPEG 扫描件不旋转的页可以按块对齐平移、几乎无损地重存，旋转的页则要完整地重新压缩一次，
    画质和体积都吃亏。而 0.2°～0.3° 以内的倾斜肉眼本来就看不出来，不值得为它旋转。
    返回 (建议值, 理由, [(角度, 倾斜不小于该角度的页数), ...])；没有可修正的页时返回 None。
    """
    pages = [p for p in infos if p.bbox]
    if not pages:
        return None
    mostly, bilevel = _mostly_bilevel(infos)
    if mostly:
        dpi = float(np.median([p.dpi for p in bilevel]))
        value = 0.3 if dpi < 250 else 0.2
        reason = f"黑白二值扫描、约 {dpi:.0f} DPI，旋转会让笔画边缘变糙，看不出来的轻微倾斜不值得旋转"
    elif profile and profile["jpeg"]:
        value = 0.2
        reason = "JPEG 扫描：不旋转的页可以按块对齐平移，几乎无损、体积也小；旋转的页要重新压缩，体积约多三成"
    else:
        value = 0.1
        reason = "无损压缩的灰度/彩色扫描，旋转几乎无损，轻微的倾斜也可以修正"
    angles = np.abs([p.angle for p in pages])
    counts = [(t, int((angles >= t - 1e-9).sum())) for t in ANGLE_STEPS]
    return value, reason, counts


def recommend_quality(infos, profile=None):
    """给「JPEG 质量」提一个建议值。返回 (建议值或 None, 理由)；None 表示这一项对本书不起作用。

    原文件是 JPEG 时，它的质量就是画质的上限：用更高的质量重存，文件成倍变大（质量 62 的原图
    用 90 重存是 2.7 倍），细节却不会比原图多。所以建议值取「原图质量略高一档」——高出的这一点
    用来抵消二次压缩带来的损失。
    """
    if not any(p.bbox for p in infos):
        return None, ""
    if _mostly_bilevel(infos)[0]:
        return None, "本书输出为 1bit 黑白图像，不使用 JPEG，「JPEG 质量」不起作用"
    if profile and profile["jpeg"]:
        q = profile["quality"]
        value = int(min(max(math.ceil((q + 5) / 5) * 5, 50), 95))
        return value, (f"原文件是 JPEG、质量约 {q:.0f}，这就是画质的上限：用更高的质量重存只会让文件变大；"
                       f"取略高一档用来抵消二次压缩的损失")
    return 90, "原文件是无损压缩的图像，用较高的质量保留细节"


def recommend_max_angle(infos, max_angle):
    """给「倾斜检测范围」提建议。max_angle 是产生 infos 的那次分析用的范围。返回 (建议值, 理由)。

    检测只在 ±max_angle 以内搜索。有页面的结果顶在范围的边上，说明它实际可能更歪、只是被范围
    截住了，应该放宽了重新分析；否则现在的范围就够用——缩小它没有好处（只是分析快一点，
    但改了这一项就得重新分析一遍）。
    """
    angles = np.abs([p.angle for p in infos if p.bbox])
    if not len(angles):
        return None, ""
    at_limit = int((angles >= max_angle - 0.26).sum())
    if at_limit:
        value = float(min(max_angle + 3, 15))
        if value > max_angle:
            return value, (f"有 {at_limit} 页检测到的倾斜顶在了 ±{max_angle:g}° 的边上，实际可能更歪；"
                           f"放宽范围后需要重新分析")
    return float(max_angle), f"全书最大的倾斜是 {angles.max():.2f}°，±{max_angle:g}° 的范围足够"


def recommend_dpi(profile):
    """给「渲染 DPI」提建议。返回 (建议值或 None, 理由)；None 表示这一项对本书不起作用。

    渲染 DPI 只用于取不了内嵌原图、需要渲染的页。按这些页里图像本身的分辨率渲染最合适：
    低了丢细节，高了只是把同样的像素放大，文件变大、清晰度不变。
    """
    if not profile:
        return None, ""
    if not profile["render_pages"]:
        return None, "全部页都直接取用内嵌原图，「渲染 DPI」不起作用"
    if not profile["render_dpi"]:
        return None, ""
    value = int(min(max(round(profile["render_dpi"] / 50) * 50, 100), 600))
    return value, (f"有 {profile['render_pages']} 页取不了内嵌原图、需要渲染，这些页里的图像约 "
                   f"{profile['render_dpi']:.0f} DPI：按图像本身的分辨率渲染，不丢细节也不虚增体积；"
                   f"改了这一项需要重新分析")


def recommend_flatten(infos):
    """给「纸面找平」（漫画）提建议。返回 (True / False / None, 理由)；None 表示这一项对本书不起作用。"""
    pages = [p for p in infos if p.bbox and not p.bilevel]
    if not pages:
        return None, "本书没有灰度/彩色页，「纸面找平」不起作用"
    shades = np.array([p.shade for p in pages])
    shaded = shades[shades >= FLATTEN_MIN_SHADE]
    if len(shaded) >= max(3, 0.1 * len(pages)):
        return True, (f"{len(pages)} 页灰度/彩色页里有 {len(shaded)} 页的页边纸色比中部暗 {FLATTEN_MIN_SHADE} 级以上"
                      f"（中位 {np.median(shaded):.0f} 级，多半是订口阴影），找平后页面均匀、页与页观感一致，"
                      f"墨线和网点相对纸面的深浅不变")
    return False, f"纸面已经均匀（页边和中部的纸色最多差 {shades.max():.0f} 级），找平没有明显的改善"


def format_recommendation(infos, profile=None, max_angle=5.0, book="text"):
    """把建议整理成 {"min_angle", "quality", "max_angle", "dpi", "flatten": 值或 None, "lines": [几行说明]}。

    命令行和界面共用。max_angle 是产生 infos 的那次分析用的倾斜检测范围。book 是书的类型：
    漫画多一项「纸面找平」的建议，文字书没有（这一项对文字书不显示）。
    """
    empty = {"min_angle": None, "quality": None, "max_angle": None, "dpi": None, "flatten": None, "lines": []}
    rec = recommend_min_angle(infos, profile)
    if rec is None:
        return empty
    value, reason, counts = rec
    total = sum(1 for p in infos if p.bbox)
    lines = ["倾斜分布：" + "  ".join(f"≥{t}°: {n} 页" for t, n in counts) + f"  （共 {total} 页）"]
    rotated = dict(counts)[value]
    lines.append(f"「小于此角度不旋转」建议设为 {value}°：{reason}。")
    lines.append(f"  按此设置有 {rotated} 页需要旋转，其余 {total - rotated} 页只平移、不旋转。")
    quality, q_reason = recommend_quality(infos, profile)
    if quality is not None:
        lines.append(f"「JPEG 质量」建议设为 {quality}：{q_reason}。")
    elif q_reason:
        lines.append(q_reason + "。")
    wide, w_reason = recommend_max_angle(infos, max_angle)
    if wide is not None and wide != max_angle:
        lines.append(f"「倾斜检测范围」建议设为 ±{wide:g}°：{w_reason}。")
    elif w_reason:
        lines.append(f"「倾斜检测范围」不用改：{w_reason}。")
    dpi, d_reason = recommend_dpi(profile)
    if dpi is not None:
        lines.append(f"「渲染 DPI」建议设为 {dpi}：{d_reason}。")
    elif d_reason:
        lines.append(d_reason + "。")
    flatten = None
    if book == "manga":
        flatten = recommend_flatten(infos)[0]
        lines.append(flatten_summary(infos))
    else:
        lines.append(enhancement_summary(infos))
    return {"min_angle": value, "quality": quality, "max_angle": wide, "dpi": dpi, "flatten": flatten, "lines": lines}


def plan_page(info, ref, opts, skip=False):
    """根据分析结果和选项，决定这一页实际要做的旋转角度和平移量。

    skip 表示用户指定了「本页不修正」：不旋转、不平移，但它的分析结果仍然参与全书标准版心的统计。
    「不修正」管的是位置，不挡「去除边缘污染」：封面、插图页这类不想让程序挪动的页恰恰常有黑边。
    同时指定了去污的页不能原样复制，而是位置不动（像素原样搬运）、只去污。
    """
    if skip:
        if cleanup_box(info, ref) is not None:
            return 0.0, (0.0, 0.0), False, "本页不修正（手动指定）  去除边缘污染"
        return 0.0, (0.0, 0.0), True, "本页不修正（手动指定）"
    if info.full_bleed_color and opts.book == "text":
        # 文字书里的彩色封面、整页彩图：原样保留（漫画整页都是画面，出血的彩页也照常纠偏、对齐）。
        # 和「本页不修正」一样，用户逐页指定的去污照做
        if cleanup_box(info, ref) is not None:
            return 0.0, (0.0, 0.0), False, "彩色整页图片，位置不动  去除边缘污染"
        return 0.0, (0.0, 0.0), True, "彩色整页图片，原样保留"
    angle = info.angle if opts.deskew and abs(info.angle) >= opts.min_angle else 0.0
    shift = (0.0, 0.0)
    if info.bbox and opts.center:
        shift = compute_shift(info, ref, opts.per_page)
    trivial = angle == 0.0 and max(abs(shift[0]), abs(shift[1])) < 0.003
    cleanup = cleanup_box(info, ref) is not None
    method = enhancement_of(info, opts)
    flat = flatten_of(info, opts)
    untouched = info.mode == "copy" or (trivial and not opts.clean_margin and not cleanup and not method and not flat)
    if untouched:
        status = info.note or "无需修正"
    else:
        status = (f"旋转 {angle:+.2f}°  平移 x{shift[0] * 100:+.1f}% y{shift[1] * 100:+.1f}%"
                  + ("  去除边缘污染" if cleanup else "")
                  + (f"  显示增强: {describe_enhancement(method)}" if method else "")
                  + (f"  纸面找平（灰影 {info.shade:.0f} 级）" if flat else "")
                  + (f"  ({info.note})" if info.note else ""))
    return angle, shift, untouched, status


def process_document(doc, infos, opts, output, progress=None, log=None, cancelled=None,
                     copy_toc=True, ref=None, skip_pages=(), delete_pages=()):
    """第 2 遍：变换并输出。返回实际修正的页数。

    ref 是标准版心的统计结果；只处理部分页时可传入全书的统计，省略则从 infos 计算。
    skip_pages 是用户指定「不做修正」的页（从 0 开始的页序）。
    delete_pages 是用户指定删除的页：不输出到新 PDF（原文件不受影响），目录书签的页码相应前移。
    """
    opts = opts.effective()
    ref = ref or compute_reference(infos)
    kept = [info.index for info in infos if info.index not in delete_pages]
    if not kept:
        raise ValueError("要处理的页全部被标记为删除，没有可输出的页")
    out = fitz.open()
    changed = 0
    for k, info in enumerate(infos, 1):
        if cancelled and cancelled():
            raise Cancelled
        if info.index in delete_pages:
            if log:
                log(f"[{k}/{len(infos)}] 第 {info.index + 1} 页: 已删除，不输出")
            if progress:
                progress(k, len(infos))
            continue
        page = doc[info.index]
        angle, shift, untouched, status = plan_page(info, ref, opts, skip=info.index in skip_pages)
        if untouched:
            out.insert_pdf(doc, from_page=info.index, to_page=info.index)
        else:
            img, method = render_page(doc, info, ref, opts, angle, shift, skip=info.index in skip_pages)
            add_output_page(out, page.rect, img, info, opts, method)
            changed += 1
        if log:
            log(f"[{k}/{len(infos)}] 第 {info.index + 1} 页: {status}")
        if progress:
            progress(k, len(infos))

    if copy_toc:
        toc = remap_toc(doc.get_toc(), kept)
        if toc:
            out.set_toc(toc)
    out.set_metadata(doc.metadata)
    out.save(output, garbage=3, deflate=True)
    return changed


def page_source_bytes(doc, page):
    """这一页内嵌图像在原 PDF 里占的字节数（压缩后的流长度）。"""
    total = 0
    for item in page.get_images(full=True):
        if doc.xref_get_key(item[0], "Filter")[0] == "null":
            # 没压缩过的流：输出保存时会顺带压缩它，按压缩之后的大小算（否则原样复制的页会被高估好几倍）
            total += len(zlib.compress(doc.xref_stream_raw(item[0]), 6))
            continue
        kind, value = doc.xref_get_key(item[0], "Length")
        if kind == "int":
            total += int(value)
    return total


def estimate_output_size(doc, infos, ref, opts, skip_pages=(), delete_pages=(), samples=12,
                         cache=None, cancelled=None):
    """预估输出文件的大小（字节）。返回 (预估值, 原样复制的页数, 重新编码的页数, 抽样页数)。

    原样复制的页按它在原 PDF 里的图像大小算。要重新编码的页，均匀抽 samples 页真的处理一遍，
    得到「处理后大小 / 原图大小」的比例，再用这个比例推算其余的页——页与页之间内容多少差别
    很大，按比例推比按平均每页大小推准得多。cache 用来在多次预估之间复用抽样页的编码结果。

    抽样页的大小不能直接用编码出来的图像字节数：PNG 交给 PyMuPDF 之后会被解码、重新压缩成
    PDF 自己的流，1bit 黑白页因此会小 25% 左右。所以把抽样页真的写进一个内存里的临时 PDF，
    按和正式输出相同的方式保存，以它的实际大小为准。
    """
    opts = opts.effective()
    cache = cache if cache is not None else {}
    copied_bytes, copied, work = 0, 0, []
    for info in infos:
        if info.index in delete_pages:
            continue
        angle, shift, untouched, _ = plan_page(info, ref, opts, skip=info.index in skip_pages)
        src = page_source_bytes(doc, doc[info.index])
        if untouched:
            copied_bytes += src
            copied += 1
        else:
            work.append((info, angle, shift, src))
    if not work:
        return copied_bytes + 1500 * copied, copied, 0, 0

    # 旋转的页和只平移的页分开抽样、分开算比例：只平移的页（尤其是块对齐的 JPEG）重存后几乎
    # 不变大，旋转的页要大三到五成，混在一起抽样，预估会随两类页的比例飘
    total, n_sampled = copied_bytes + 1500 * copied, 0
    # 做了显示增强的页（放大 2、3 倍，或者变成矢量）和别的页也不是一个比例，同样要分开
    groups = {}
    for item in work:
        skip = item[0].index in skip_pages
        groups.setdefault((item[1] == 0.0, enhancement_of(item[0], opts, skip), flatten_of(item[0], opts, skip)),
                          []).append(item)
    for group in groups.values():
        n = min(len(group), max(3, round(samples * len(group) / len(work))))
        step = len(group) / n
        picked = sorted({int(k * step) for k in range(n)})
        tmp = fitz.open()
        for j in picked:
            if cancelled and cancelled():
                raise Cancelled
            info, angle, shift, src = group[j]
            skip = info.index in skip_pages
            key = (info.index, round(angle, 3), round(shift[0], 4), round(shift[1], 4), opts.clean_margin,
                   opts.upscale, None if info.bilevel else opts.quality, opts.dpi, cleanup_box(info, ref),
                   enhancement_of(info, opts, skip), flatten_of(info, opts, skip), skip)
            if key not in cache:                    # 缓存一页单独存成 PDF 的字节，多次预估之间复用
                single = fitz.open()
                img, method = render_page(doc, info, ref, opts, angle, shift, skip=skip)
                add_output_page(single, doc[info.index].rect, img, info, opts, method)
                cache[key] = single.tobytes(garbage=3, deflate=True)
            with fitz.open("pdf", cache[key]) as single:
                tmp.insert_pdf(single)

        sampled_src = sum(group[j][3] for j in picked)
        sampled_out = len(tmp.tobytes(garbage=3, deflate=True))    # 已含这几页的 PDF 结构开销
        rest = [w[3] for j, w in enumerate(group) if j not in picked]
        if sampled_src > 0:
            total += sampled_out + sum(rest) * sampled_out / sampled_src
        else:                               # 原图大小取不到（渲染模式等）：退回到按平均每页大小推算
            total += sampled_out + len(rest) * sampled_out / len(picked)
        total += 1500 * len(rest)           # 1500: 每页的 PDF 结构开销
        n_sampled += len(picked)
    return int(total), copied, len(work), n_sampled


def format_size(n):
    return f"{n / 1048576:.1f} MB" if n >= 1048576 else f"{n / 1024:.0f} KB"


def remap_toc(toc, kept):
    """删掉一些页之后，把目录书签的页码换算成新 PDF 里的页码。

    kept 是保留下来的页（原页序，从 0 开始，升序）。指向被删页的书签改为指向它后面
    第一个保留页（后面没有了就指向最后一页）；书签本身不删，免得破坏目录的层级。
    """
    result = []
    for entry in toc:
        entry = list(entry)
        if entry[2] >= 1:
            j = min(bisect.bisect_left(kept, entry[2] - 1), len(kept) - 1)
            entry[2] = j + 1
        result.append(entry)
    return result


def preview_page(doc, info, ref, opts, skip=False):
    """生成单页的处理前/处理后图像（供图形界面预览）。"""
    page = doc[info.index]
    opts = opts.effective()
    angle, shift, untouched, status = plan_page(info, ref, opts, skip)
    render_info = info if info.mode != "copy" else PageInfo(index=info.index, mode="render")
    before = load_page_image(doc, page, render_info, 100 if info.mode == "copy" else opts.dpi)
    if untouched:
        after = before
    else:
        after, method = render_page(doc, info, ref, opts, angle, shift, skip=skip)
        if "vector" in method:              # 矢量化的页：把实际写出来的那一页渲染成图，所见即所得
            with fitz.open() as tmp:
                zoom = after.shape[1] / page.rect.width
                pix = add_output_page(tmp, page.rect, after, info, opts, method).get_pixmap(
                    matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csGRAY)
                after = pixmap_to_array(pix)
        elif info.bilevel:                  # 预览也按实际输出那样二值化
            after = cv2.threshold(to_gray(after), 127, 255, cv2.THRESH_BINARY)[1]
    box = None
    if info.bbox:       # 不动的页（不修正、无需修正）也画出红框：去污是按红框算的，得让用户先看到它
        b = page_box(info, ref)
        box = (b[0] + shift[0], b[1] + shift[1], b[2] + shift[0], b[3] + shift[1])
    return before, after, box, status


def parse_pages(spec, total):
    if not spec:
        return list(range(total))
    pages = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        start, end = int(a), int(b) if b else int(a)
        pages.extend(range(max(start, 1) - 1, min(end, total)))
    return pages


def main():
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(errors="replace")

    ap = argparse.ArgumentParser(description="扫描书籍 PDF 整形：纠偏 + 版心居中")
    ap.add_argument("input", type=Path)
    ap.add_argument("-o", "--output", type=Path)
    ap.add_argument("--pages", help="只处理指定页，如 1-20 或 3,5,8-12（页码从 1 开始）")
    ap.add_argument("--skip-pages", help="这些页不做修正、原样保留，写法同 --pages")
    ap.add_argument("--delete-pages", help="这些页不输出到新 PDF，写法同 --pages")
    ap.add_argument("--cleanup-pages", help="这些页去除版心以外的边缘污染，写法同 --pages")
    ap.add_argument("--max-angle", type=float, default=5.0, help="倾斜检测范围 ±度 (默认 5)")
    ap.add_argument("--min-angle", type=float, default=None,
                    help="小于此角度不旋转 (默认: 分析全书后自动采用建议值)")
    ap.add_argument("--dpi", type=int, default=300, help="无法直接取原图的页面的渲染 DPI (默认 300)")
    ap.add_argument("--quality", type=int, default=None,
                    help="JPEG 质量 (默认: 分析全书后自动采用建议值；原文件不是 JPEG 时为 90)")
    ap.add_argument("--no-deskew", action="store_true", help="不做倾斜校正")
    ap.add_argument("--no-center", action="store_true", help="不做版心居中")
    ap.add_argument("--per-page", action="store_true",
                    help="每页各自居中（默认会参照全书标准版心，避免半页内容跑到页面中间）")
    ap.add_argument("--clean-margin", action="store_true", help="把版心以外涂成纸色（去黑边、阴影）")
    ap.add_argument("--enhance", action="store_true",
                    help="显示增强：对能有明显改善的黑白页，自动选合适的方法（平滑放大、去噪点、矢量化）")
    ap.add_argument("--upscale", action="store_true",
                    help="黑白二值页旋转时以 2 倍分辨率输出（笔画边缘更平滑，体积约 2.7 倍）")
    ap.add_argument("--book", choices=list(BOOK_TYPES), default="text",
                    help="书的类型：text 文字书（默认）/ manga 漫画书。漫画可用 --flatten，不用 --enhance/--upscale")
    ap.add_argument("--flatten", action="store_true",
                    help="纸面找平（漫画）：把页边的灰影（订口阴影）按当地纸色拉白，只对有明显灰影的灰度页")
    args = ap.parse_args()

    output = args.output or default_output_path(args.input)
    opts = Options(book=args.book, deskew=not args.no_deskew, center=not args.no_center, per_page=args.per_page,
                   clean_margin=args.clean_margin, upscale=args.upscale, enhance=args.enhance, flatten=args.flatten,
                   max_angle=args.max_angle, min_angle=args.min_angle or 0.0, dpi=args.dpi, quality=args.quality or 90)
    doc = fitz.open(args.input)
    indices = parse_pages(args.pages, doc.page_count)

    infos = analyze_document(
        doc, indices, opts,
        progress=lambda k, n: print(f"\r分析中 {k}/{n}", end="", flush=True))
    print()
    rec = format_recommendation(infos, source_profile(doc, infos), opts.max_angle, book=opts.book)
    for line in rec["lines"]:
        print(line)
    if args.min_angle is None:
        opts.min_angle = rec["min_angle"] if rec["min_angle"] is not None else 0.1
        print(f"未指定 --min-angle，采用 {opts.min_angle}°")
    if args.quality is None and rec["quality"] is not None:
        opts.quality = rec["quality"]
        print(f"未指定 --quality，采用 {opts.quality}")
    skip_pages = set(parse_pages(args.skip_pages, doc.page_count)) if args.skip_pages else set()
    delete_pages = set(parse_pages(args.delete_pages, doc.page_count)) if args.delete_pages else set()
    ref = compute_reference(infos)
    if ref and args.cleanup_pages:
        ref["cleanup"] = set(parse_pages(args.cleanup_pages, doc.page_count))
    estimate = estimate_output_size(doc, infos, ref, opts, skip_pages, delete_pages)[0]
    print(f"预计输出约 {format_size(estimate)}（原文件 {format_size(args.input.stat().st_size)}）")
    changed = process_document(doc, infos, opts, output, log=print, copy_toc=not args.pages, ref=ref,
                               skip_pages=skip_pages, delete_pages=delete_pages)
    deleted = sum(1 for p in infos if p.index in delete_pages)
    print(f"\n完成：共 {len(infos)} 页，修正 {changed} 页"
          + (f"，删除 {deleted} 页" if deleted else "") + f" → {output}（{format_size(Path(output).stat().st_size)}）")


if __name__ == "__main__":
    main()
