"""测量输出 PDF 每页的残余倾斜和版心位置。"""
import os, sys
import cv2, fitz, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import pdf_reshape as pr

doc = fitz.open(sys.argv[1])
for i, page in enumerate(doc):
    info = pr.PageInfo(index=i)
    info.mode, info.xref = pr.classify_page(doc, page)
    if info.mode == "copy":
        info.mode = "render"
    img = pr.load_page_image(doc, page, info, 150)
    gray = pr.to_gray(img)
    s = pr.ANALYSIS_LONG_SIDE / max(gray.shape)
    ink = pr.make_ink_mask(cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
    ang, conf, _ = pr.detect_skew(ink, 5)
    b = pr.detect_content_bbox(ink)[0]
    print(f"p{i+1}: residual={ang:+.2f}  left={b[0]:.3f} right={1-b[2]:.3f}  top={b[1]:.3f} bottom={1-b[3]:.3f}")
