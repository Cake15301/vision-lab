# 图像批处理工具（day32.py）

把一个文件夹里的图批量处理成统一规格：按宽度等比缩放、转灰度、拉对比度。
每张图同时生成一张"原图 / 处理后"的并排对比图，并输出一份亮度体检表。

## 怎么跑

在 `practice` 文件夹里（脚本和素材都在这里）：

```bash
D:/python/python.exe day32.py
```

不带参数时用默认设置：宽度 800、对比度 1.2、太暗门槛 55。

看每个参数是什么意思：

```bash
D:/python/python.exe day32.py -h
```

改参数跑，例如宽度改成 400、对比度拉到 1.5：

```bash
D:/python/python.exe day32.py --width 400 --alpha 1.5
```

## 参数

| 参数 | 默认值 | 作用 |
|---|---|---|
| `--in-dir` | `D:/AI/zcode files/vision-lab/practice/day31_assets/in` | 原料文件夹 |
| `--out-dir` | `D:/AI/zcode files/vision-lab/practice/day31_assets/out` | 成品文件夹 |
| `--width` | `800` | 成品统一宽度，单位像素 |
| `--alpha` | `1.2` | 对比度倍数，1.0 表示不变 |
| `--too-dark` | `55` | 平均亮度低于它判"太暗" |

## 输出什么

- `out/`：处理后的灰度图，文件名和原图一样，方便追溯是哪张处理的
- `out/cmp/`：并排对比图，左边原图、右边处理后，宽度是成品的两倍
- `out/体检报告.txt`：每张一行的亮度与判定（太暗 / 正常 / 过曝）

## 注意

- 太暗门槛 55 是按这批样本的光照定的。换一套光源要重新标定，不能照搬。
- 跑之前先确认站在 `practice` 文件夹里，并且用的是 `D:/python/python.exe`。