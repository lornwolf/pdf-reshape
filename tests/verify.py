#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测量一个 PDF 每页的残余倾斜和四边边距（都按页面宽高归一化）。

用法:  python tests/verify.py 输出.pdf

处理得好的页：residual ≈ 0（±0.05 是搜索网格的误差），left ≈ right，top ≈ bottom。
"""
import os
import sys

import cv2
import fitz

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import pdf_reshape as pr


def measure(path):
    """返回 [{"angle", "left", "right", "top", "bottom"}, ...]，每页一项；空白页的边距为 None。"""
    result = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc):
            info = pr.PageInfo(index=i)
            info.mode, info.xref = pr.classify_page(doc, page)
            if info.mode == "copy":
                info.mode = "render"
            gray = pr.to_gray(pr.load_page_image(doc, page, info, 150))
            s = pr.ANALYSIS_LONG_SIDE / max(gray.shape)
            ink = pr.make_ink_mask(cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
            angle = pr.detect_skew(ink, 5)[0]
            box = pr.detect_content_bbox(ink)[0]
            if box is None:
                result.append({"angle": angle, "left": None, "right": None, "top": None, "bottom": None})
            else:
                result.append({"angle": angle, "left": box[0], "right": 1 - box[2],
                               "top": box[1], "bottom": 1 - box[3]})
    return result


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    for i, m in enumerate(measure(sys.argv[1]), 1):
        if m["left"] is None:
            print(f"p{i}: residual={m['angle']:+.2f}  （没有检测到内容）")
        else:
            print(f"p{i}: residual={m['angle']:+.2f}  left={m['left']:.3f} right={m['right']:.3f}"
                  f"  top={m['top']:.3f} bottom={m['bottom']:.3f}")
