# -*- coding: utf-8 -*-
"""测试的公共部分：工作目录、测试文件、一个极简的运行器（不依赖 pytest）。"""
import os
import sys
import tempfile
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

for _stream in (sys.stdout, sys.stderr):        # 控制台不一定是 UTF-8 代码页
    _stream.reconfigure(errors="replace")

# 所有测试产生的文件都放在这里（系统临时目录下），不碰项目目录，也不碰用户真实的进度数据库。
# run_all.py 会把它通过环境变量传给各个子进程，让测试文件只生成一次。
WORK = os.environ.get("PDF_RESHAPE_TEST_DIR") or tempfile.mkdtemp(prefix="pdf_reshape_test_")
os.environ["PDF_RESHAPE_TEST_DIR"] = WORK


def work_path(name):
    return os.path.join(WORK, name)


def fixtures():
    """生成（或取用已生成的）合成测试 PDF，返回 {文件名: 路径}。"""
    from make_test import build_all
    return build_all(WORK)


class Failures(list):
    """收集一组检查里没通过的项，全部检查完再一起报告，而不是遇到第一个就停。"""

    def check(self, condition, message):
        if not condition:
            self.append(message)
        return bool(condition)

    def close(self, a, b, tolerance, message):
        return self.check(a is not None and b is not None and abs(a - b) <= tolerance,
                          f"{message}：{a} 与 {b} 相差超过 {tolerance}")


def run_tests(tests):
    """依次运行 [(名字, 函数)]；函数返回 Failures（或抛异常）。返回没通过的个数。"""
    failed = 0
    for name, fn in tests:
        try:
            problems = fn() or []
        except Exception:
            problems = [traceback.format_exc().rstrip()]
        if problems:
            failed += 1
            print(f"FAIL  {name}")
            for p in problems:
                print("      " + str(p).replace("\n", "\n      "))
        else:
            print(f"ok    {name}")
    return failed
