import os, sys, random
import cv2, fitz, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import pdf_reshape as pr

W, H = 1240, 1754
random.seed(1)

def make_page(lines, angle, dx, dy, black_edge, vertical=False):
    img = np.full((H, W), 235, np.uint8)
    for r in range(lines):
        y = 300 + r * 42
        x = 220
        while x < 1000:
            word = "".join(random.choice("abcdefghmnopqrstuw") for _ in range(random.randint(2, 9)))
            cv2.putText(img, word, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 30, 2, cv2.LINE_AA)
            x += cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0][0] + 18
    if vertical:
        img = cv2.resize(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)[:, :], (W, H))
    m = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
    m[0, 2] += dx; m[1, 2] += dy
    img = cv2.warpAffine(img, m, (W, H), borderValue=235)
    if black_edge:
        img[:, :35] = 15
        img[:25, :] = 15
    return img

specs = [  # lines, angle, dx, dy, black_edge, vertical
    (28, 2.3, 120, -90, True, False),
    (28, -1.7, -150, 60, False, False),
    (28, 0.0, 0, 0, False, False),
    (28, 3.8, 80, 140, True, False),
    (28, -0.6, -100, -120, False, False),
    (8, 1.2, 130, -70, False, False),     # 章末半页：应贴上边，而不是垂直居中
    (28, -2.9, 60, 30, True, False),
    (28, 1.5, -60, 90, False, False),
]
doc = fitz.open()
for s in specs:
    img = make_page(*s)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    p = doc.new_page(width=595, height=842)
    p.insert_image(p.rect, stream=buf.tobytes(), keep_proportion=False)
doc.save(sys.argv[1])

# 竖排测试
doc = fitz.open()
for s in [(28, 2.0, 100, 50, False, True), (28, -3.1, -80, -60, True, True),
          (28, 0.8, 40, -100, False, True), (28, -1.2, -120, 80, False, True)]:
    img = make_page(*s)
    ok, buf = cv2.imencode(".jpg", img)
    p = doc.new_page(width=595, height=842)
    p.insert_image(p.rect, stream=buf.tobytes(), keep_proportion=False)
doc.save(sys.argv[2])
