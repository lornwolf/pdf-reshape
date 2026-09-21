#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""一键运行全部回归测试：先核心、后界面。

用法:
    python tests/run_all.py             # 全部（约 2 分钟；界面测试会弹出窗口自己操作，期间别碰它）
    python tests/run_all.py --no-gui    # 只跑核心测试（约 1 分钟，不弹窗口）

测试只用合成的 PDF（tests/make_test.py 生成），所有文件和数据库都在系统临时目录下。
用真实的扫描书核对另见 tests/check_book.py。
"""
import shutil
import sys
import time

import harness
import test_core
import test_gui


def main():
    started = time.time()
    print(f"工作目录: {harness.WORK}")
    harness.fixtures()
    failed = harness.run_tests(test_core.TESTS)
    if "--no-gui" not in sys.argv:
        failed += test_gui.run_all([])
    print(f"\n{'全部通过' if not failed else f'有 {failed} 项没通过'}，用时 {time.time() - started:.0f} 秒")
    if not failed:
        shutil.rmtree(harness.WORK, ignore_errors=True)     # 没通过时留着现场，方便查
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
