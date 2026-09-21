# 回归测试

不依赖 pytest，直接用 Python 运行。所有测试产生的文件和数据库都在系统临时目录下，
不会碰项目目录，也不会碰你真实的校正进度（`~/.pdf_reshape/books.db`）。

## 一键运行

```
python tests/run_all.py             # 全部，约 2 分钟
python tests/run_all.py --no-gui    # 只跑核心测试，约 1 分钟，不弹窗口
```

界面测试会弹出程序窗口并自己操作（点按钮、翻页、处理输出），**期间不要动鼠标键盘去碰它**。
全部通过时临时文件会被清掉；有没通过的，工作目录会留着（开头打印了路径），方便查。

## 各个文件

| 文件 | 作用 |
|---|---|
| `run_all.py` | 一键运行：先核心、后界面 |
| `test_core.py` | 核心处理和进度数据库的测试，不需要界面。`python tests/test_core.py [名字的一部分 ...]` 可以只跑几个 |
| `test_gui.py` | 界面测试，14 个场景，每个场景一个子进程、带超时。`python tests/test_gui.py marks keys` 只跑几个；`--run 名字` 在当前进程里跑（调试用） |
| `gui_driver.py` | 驱动界面的工具：场景写成生成器，要等的时候 `yield` 一个条件 |
| `make_test.py` | 生成合成的测试 PDF（每页的倾斜和偏移都是已知的）：横排、竖排、图像蒙版、大倾斜 + 需要渲染的页 |
| `verify.py` | 测量一个 PDF 每页的残余倾斜和四边边距。`python tests/verify.py 输出.pdf` |
| `check_book.py` | **用真实的扫描书核对**，见下 |
| `harness.py` | 公共部分：工作目录、测试文件、极简的运行器 |

## 用真实的书核对

合成数据覆盖不了真实扫描件的边缘杂质和各种版式（页码挂在正文外面、整体缩进的页、左右页位置不同……）。
**改了版心检测或对齐规则之后，必须拿真实的书整本过一遍：**

```
python tests/check_book.py 书.pdf                  # 统计 + 可疑的页
python tests/check_book.py 书.pdf --pages 9,27     # 另外打印这几页的细节
python tests/check_book.py 书.pdf --process        # 再整本处理一遍，对比预估大小和实际大小
```

全书分析要几分钟，结果会缓存（按文件指纹 + 算法版本）；只改对齐规则时几秒钟就能看到结果。
怎么看结果写在 `check_book.py` 开头；各本基准样本应有的结果记在项目根目录的 `CLAUDE.md` 里。

## 加新测试

- 核心逻辑：在 `test_core.py` 里加一个 `test_` 开头的函数，返回 `Failures`（用 `f.check(条件, 说明)` /
  `f.close(a, b, 容差, 说明)` 记录问题，全部检查完一起报告）。
- 界面：在 `test_gui.py` 里写一个场景并登记到 `SCENARIOS`。要重启程序才能验证的（保存与恢复），
  拆成两个场景、共用一个数据库，参见 `marks` 和 `marks_restore`。
- 修 bug 时先写一个能复现它的测试，再改代码。
