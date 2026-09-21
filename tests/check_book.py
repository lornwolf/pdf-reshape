#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""用一本真实的扫描书核对版心检测和对齐规则。合成数据覆盖不了真实扫描件的边缘杂质和各种版式，
改了这两块之后必须拿真实的书整本过一遍。

用法:
    python tests/check_book.py 书.pdf                  # 分析全书，打印统计和可疑的页
    python tests/check_book.py 书.pdf --pages 9,27     # 另外打印这几页的细节
    python tests/check_book.py 书.pdf --process        # 再整本处理一遍，对比预估大小和实际大小
    python tests/check_book.py 书.pdf --fresh          # 不用缓存，重新分析

全书分析要几分钟，结果按「文件指纹 + 算法版本」缓存在临时目录里；只改对齐规则（不改检测）时
不用重新分析，几秒钟就能看到结果。

怎么看结果：
    - 「标准版心」的宽高要和这本书正文的实际大小一致，宽度分布要集中；
    - 「单侧对齐」的页应该是整体缩进的页（目录、练习题、字表），「其他」应该很少；
    - 「照搬邻页」的变动要小：中位数 0.3% 左右，最大不超过 3%；
    - 逐个看「可疑的页」：版心比标准明显大的、平移量特别大的。
"""
import argparse
import os
import pickle
import sys
import tempfile
import time

import fitz
import numpy as np

import harness  # noqa: F401  （把项目目录加进导入路径）
import pdf_reshape as pr
from pdf_reshape_store import fingerprint


# 缓存放在固定的目录里（不是每次运行都新建的测试工作目录），这样下次运行还能用上
CACHE_DIR = os.path.join(tempfile.gettempdir(), "pdf_reshape_check_book")


def load_infos(path, fresh):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"analysis_{fingerprint(path)[:16]}_v{pr.ANALYSIS_VERSION}.pkl")
    if not fresh and os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)
    started = time.time()
    with fitz.open(path) as doc:
        infos = pr.analyze_document(
            doc, range(doc.page_count), pr.Options(),
            progress=lambda k, n: print(f"\r分析中 {k}/{n}", end="", flush=True))
    print(f"\r分析完成，用时 {time.time() - started:.0f} 秒" + " " * 20)
    with open(cache, "wb") as f:
        pickle.dump(infos, f)
    return infos


def main():
    ap = argparse.ArgumentParser(description="用一本真实的扫描书核对版心检测和对齐规则")
    ap.add_argument("pdf")
    ap.add_argument("--pages", help="另外打印这些页的细节，如 9,27,50")
    ap.add_argument("--process", action="store_true", help="整本处理一遍，对比预估大小和实际大小")
    ap.add_argument("--fresh", action="store_true", help="不用缓存，重新分析")
    args = ap.parse_args()

    infos = load_infos(args.pdf, args.fresh)
    ref = pr.compute_reference(infos)
    pages = [p for p in infos if p.bbox]
    axis = 1 if ref["vertical"] else 0                           # 文字行的方向：横排书是水平方向
    lo, hi = axis, axis + 2
    std = (1 - ref["ext"][axis]) / 2                            # 标准版心居中时的边距
    print(f"{os.path.basename(args.pdf)}：{len(infos)} 页，其中 {len(pages)} 页可修正，"
          f"{'竖排' if ref['vertical'] else '横排'}，{'黑白二值' if pr._mostly_bilevel(infos)[0] else '灰度/彩色'}")
    print(f"标准版心 宽 {ref['ext'][0]:.3f} 高 {ref['ext'][1]:.3f}；"
          f"两类页的左右边 {np.round([ref[0][0], ref[0][2]], 3)} / {np.round([ref[1][0], ref[1][2]], 3)}")

    width = np.array([ref["boxes"][p.index][hi] - ref["boxes"][p.index][lo] for p in pages])
    print(f"文字行方向的版心大小分布 5/25/50/75/95%: {np.round(np.percentile(width, [5, 25, 50, 75, 95]), 3)}")
    peeled = [p.index + 1 for p in pages if tuple(ref["content"][p.index]) != tuple(p.bbox)]
    dense = [p.index + 1 for p in pages if tuple(ref["boxes"][p.index]) != tuple(ref["content"][p.index])]
    print(f"剥掉了空白处杂质的页 {len(peeled)} 页: {peeled[:30]}{' …' if len(peeled) > 30 else ''}")
    print(f"有东西挂在正文外面、按行覆盖求正文的页 {len(dense)} 页")

    kinds = {"居中": [], "单侧对齐": [], "其他": []}
    shifts = []
    for p in pages:
        b = ref["boxes"][p.index]
        s = pr.compute_shift(p, ref, False)
        shifts.append(s)
        near, far = b[lo] + s[axis], 1 - b[hi] - s[axis]
        kind = "居中" if abs(near - far) < 0.004 else ("单侧对齐" if min(abs(near - std), abs(far - std)) < 0.004 else "其他")
        kinds[kind].append(p.index + 1)
    print("文字行方向上:", {k: len(v) for k, v in kinds.items()})
    print(f"   单侧对齐的页: {kinds['单侧对齐'][:40]}{' …' if len(kinds['单侧对齐']) > 40 else ''}")
    print(f"   其他: {kinds['其他'][:40]}")
    shifts = np.array(shifts)
    print("平移量 x 5/50/95%%: %s   y: %s" % (np.round(np.percentile(shifts[:, 0], [5, 50, 95]), 3),
                                          np.round(np.percentile(shifts[:, 1], [5, 50, 95]), 3)))

    big = [(p.index + 1, np.round(s, 3)) for p, s in zip(pages, shifts) if np.abs(s).max() > 0.05]
    wide = [p.index + 1 for p in pages
            if ref["boxes"][p.index][2] - ref["boxes"][p.index][0] > ref["ext"][0] + pr.OVERSIZE_TOLERANCE
            or ref["boxes"][p.index][3] - ref["boxes"][p.index][1] > ref["ext"][1] + pr.OVERSIZE_TOLERANCE]
    print(f"可疑：版心比标准大 2.5% 以上的页 {len(wide)} 页: {wide[:40]}")
    print(f"可疑：平移超过 5% 的页 {len(big)} 页: {[pg for pg, _ in big][:40]}")

    normal = {p.index for p in pages
              if abs(ref["boxes"][p.index][2] - ref["boxes"][p.index][0] - ref["ext"][0]) < 0.02
              and abs(ref["boxes"][p.index][3] - ref["boxes"][p.index][1] - ref["ext"][1]) < 0.02}
    moves = [max(abs(o) for o in pr.box_like_neighbor(infos[i], infos[i + d], ref))
             for i in normal for d in (-1, 1) if i + d in normal]
    if moves:
        print(f"照搬邻页（正常页照正常的邻页）红框的变动: 中位 {np.median(moves) * 100:.2f}%  "
              f"90% 分位 {np.percentile(moves, 90) * 100:.2f}%  最大 {max(moves) * 100:.2f}%")

    with fitz.open(args.pdf) as doc:
        rec = pr.format_recommendation(infos, pr.source_profile(doc, infos), 5.0)
        print("\n".join(rec["lines"]))

        for pg in pr.parse_pages(args.pages, len(infos)) if args.pages else []:
            p = infos[pg]
            if not p.bbox:
                print(f"第 {pg + 1} 页: {p.mode} {p.note}")
                continue
            b, s = ref["boxes"][p.index], pr.compute_shift(p, ref, False)
            print(f"第 {pg + 1} 页: 倾斜 {p.angle:+.2f}°  全部内容 {np.round(p.bbox, 3)}  对齐用 {np.round(b, 3)}"
                  f"  平移 {np.round(s, 3)}  → 处理后边距 左 {b[0] + s[0]:.3f} 右 {1 - b[2] - s[0]:.3f}"
                  f" 上 {b[1] + s[1]:.3f} 下 {1 - b[3] - s[1]:.3f}")

        if args.process:
            opts = pr.Options(min_angle=rec["min_angle"] or 0.1, quality=rec["quality"] or 90)
            estimate = pr.estimate_output_size(doc, infos, ref, opts)[0]
            out = os.path.join(CACHE_DIR, "check_book_out.pdf")
            started = time.time()
            pr.process_document(doc, infos, opts, out, ref=ref)
            actual = os.path.getsize(out)
            print(f"整本处理用时 {time.time() - started:.0f} 秒：预估 {pr.format_size(estimate)}，实际 {pr.format_size(actual)}"
                  f"（误差 {100 * (estimate - actual) / actual:+.1f}%），原文件 {pr.format_size(os.path.getsize(args.pdf))}")
            os.remove(out)


if __name__ == "__main__":
    main()
