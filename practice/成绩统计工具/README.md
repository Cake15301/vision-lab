# 成绩统计工具

一个读成绩表、算出统计结果的小工具。

## 它做什么

读一份 CSV 格式的成绩表（两列：姓名、分数），打印出：

- 全部成绩的原始数据
- 平均分
- 最高分（以及是谁考的）
- 最低分（以及是谁考的）
- 总人数
- 高于平均分的人数和名单

## 怎么装依赖

本项目不需要安装任何第三方包，只用 Python 自带的 `csv` 和 `argparse`。

依赖清单见 `requirements.txt`（目前没有外部依赖）。如果以后这个项目加了外部包，装依赖只要一条命令：

```bash
pip install -r requirements.txt
```

本项目在 Python 3.12.10 上验证通过。

（可选）如果想给这个项目单独开一个环境：

```bash
python -m venv .venv
source .venv/Scripts/activate        # Git Bash / Linux；Windows PowerShell 用 .venv\Scripts\Activate.ps1
```

## 怎么运行

成绩表格式如下，第一行必须是表头 `名字,分数`：

```csv
名字,分数
小明,85
小红,92
```

在**本项目文件夹里**运行：

```bash
python main.py --source "../成绩.csv"
```

`--source` 是必填参数，要填成绩 CSV 的路径。

运行结果（用的是仓库里 `practice/成绩.csv` 这份数据）：

```text
我是 main，我的名字是：__main__
[['小明', 85], ['小红', 92], ['小刚', 78], ['小美', 96], ['小强', 88]]
平均成绩为87.8
最高分是96,学生是小美
一共5个学生
最低分是78,学生是小刚
高于平均分的学生有3个，分别是['小红', '小美', '小强']
```

看命令帮助：

```bash
python main.py --help
```

## 项目里有什么

| 文件 | 作用 |
|---|---|
| `main.py` | 入口：解析命令行参数、组织流程、把结果打给人看 |
| `score_tools.py` | 工具函数：`read_scores` 读表、`average` 平均分、`best` 最高分、`lowest` 最低分、`count` 人数、`above_average` 高于平均分的人 |
| `requirements.txt` | 依赖清单（目前为空，本项目零第三方依赖） |