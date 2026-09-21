#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成回归测试用的合成 PDF：每一页的倾斜角度和偏移量都是已知的。

用法:
    python tests/make_test.py <输出目录>        # 生成全部四个文件
    python tests/make_test.py h.pdf v.pdf       # 旧用法：只生成横排和竖排两个

四个文件:
    h.pdf     横排 8 页（灰度 JPEG）。第 1、4、7 页带扫描黑边，第 6 页是章末半页
    v.pdf     竖排 4 页
    mask.pdf  和 h.pdf 内容相同，但每页存成 1bit 的「图像蒙版」（ImageMask，没有色彩空间）。
              有的扫描软件这样存黑白页；对它做色彩空间转换会报 source colorspace must not be None
    wide.pdf  6 页。第 2、4 页的倾斜超出默认的 ±5° 检测范围；第 3、5 页由上下两张图拼成，
              取不了内嵌原图、只能渲染
"""
import os
import random
import sys

import cv2
import fitz
import numpy as np

W, H = 1240, 1754           # A4 @ 150 DPI
PAPER, INK = 235, 30

# (行数, 倾斜角, 水平偏移, 垂直偏移, 是否带黑边)
HORIZONTAL = [
    (28, 2.3, 120, -90, True),
    (28, -1.7, -150, 60, False),
    (28, 0.0, 0, 0, False),
    (28, 3.8, 80, 140, True),
    (28, -0.6, -100, -120, False),
    (8, 1.2, 130, -70, False),      # 章末半页：应贴上边，而不是垂直居中
    (28, -2.9, 60, 30, True),
    (28, 1.5, -60, 90, False),
]
VERTICAL = [(28, 2.0, 100, 50, False), (28, -3.1, -80, -60, True),
            (28, 0.8, 40, -100, False), (28, -1.2, -120, 80, False)]
WIDE = [(1.0, 40, 20), (6.5, -30, 40), (-0.8, 20, -30), (-7.2, 30, 10), (0.4, -20, 20), (2.0, 10, 10)]
WIDE_SPLIT = (2, 4)         # wide.pdf 里拆成上下两张图的页（从 0 开始）


def make_page(rng, lines, angle, dx, dy, black_edge, vertical=False):
    img = np.full((H, W), PAPER, np.uint8)
    for r in range(lines):
        y, x = 300 + r * 42, 220
        while x < 1000:
            word = "".join(rng.choice("abcdefghmnopqrstuw") for _ in range(rng.randint(2, 9)))
            cv2.putText(img, word, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, INK, 2, cv2.LINE_AA)
            x += cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0][0] + 18
    if vertical:
        img = cv2.resize(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), (W, H))
    m = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
    m[0, 2] += dx
    m[1, 2] += dy
    img = cv2.warpAffine(img, m, (W, H), borderValue=PAPER)
    if black_edge:
        img[:, :35] = 15
        img[:25, :] = 15
    return img


def _jpeg(img, quality):
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()


def _save(pages, path, quality):
    doc = fitz.open()
    for img in pages:
        page = doc.new_page(width=595, height=842)
        page.insert_image(page.rect, stream=_jpeg(img, quality), keep_proportion=False)
    doc.save(path)


def build_horizontal(path):
    rng = random.Random(1)
    _save([make_page(rng, *spec) for spec in HORIZONTAL], path, 92)


def build_vertical(path):
    rng = random.Random(2)
    _save([make_page(rng, *spec, vertical=True) for spec in VERTICAL], path, 95)


def build_mask(path):
    rng = random.Random(1)
    doc = fitz.open()
    for spec in HORIZONTAL:
        bw = cv2.threshold(make_page(rng, *spec), 128, 255, cv2.THRESH_BINARY)[1]
        png = cv2.imencode(".png", bw, [cv2.IMWRITE_PNG_BILEVEL, 1])[1].tobytes()
        page = doc.new_page(width=595, height=842)
        xref = page.insert_image(page.rect, stream=png, keep_proportion=False)
        doc.xref_set_key(xref, "ImageMask", "true")
        doc.xref_set_key(xref, "ColorSpace", "null")
    doc.save(path)


def build_wide(path):
    rng = random.Random(3)
    doc = fitz.open()
    for k, (angle, dx, dy) in enumerate(WIDE):
        img = make_page(rng, 28, angle, dx, dy, False)
        page = doc.new_page(width=595, height=842)
        if k in WIDE_SPLIT:
            half = H // 2
            page.insert_image(fitz.Rect(0, 0, 595, 421), stream=_jpeg(img[:half], 85), keep_proportion=False)
            page.insert_image(fitz.Rect(0, 421, 595, 842), stream=_jpeg(img[half:], 85), keep_proportion=False)
        else:
            page.insert_image(page.rect, stream=_jpeg(img, 85), keep_proportion=False)
    doc.save(path)


BUILDERS = {"h.pdf": build_horizontal, "v.pdf": build_vertical, "mask.pdf": build_mask, "wide.pdf": build_wide}


def build_all(directory):
    """在 directory 里生成全部测试文件（已有的不重复生成）。返回 {文件名: 路径}。"""
    os.makedirs(directory, exist_ok=True)
    paths = {}
    for name, builder in BUILDERS.items():
        paths[name] = os.path.join(directory, name)
        if not os.path.exists(paths[name]):
            builder(paths[name])
    return paths


if __name__ == "__main__":
    if len(sys.argv) == 2:
        for name, path in build_all(sys.argv[1]).items():
            print(path)
    elif len(sys.argv) == 3:
        build_horizontal(sys.argv[1])
        build_vertical(sys.argv[2])
    else:
        sys.exit(__doc__)
