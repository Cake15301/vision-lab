# ============================================================
# day32.py —— 图像批处理工具（W05 完整版，Day32）
# 干的事：把一个文件夹里的图批量处理：
#   统一宽度缩放 → 转灰度 → 拉对比度 → 存进 out/
#   每张再拼一张"原图 | 处理后"的并排对比图，存进 out/cmp/
#   最后写一份亮度体检表 out/体检报告.txt（太暗 / 正常 / 过曝）
# 用法：
#   默认跑（800 宽、对比度 1.2、太暗门槛 55）
#   D:/python/python.exe day32.py
#   改参数跑（宽度 400、对比度 1.5、太暗门槛 58）
#   D:/python/python.exe day32.py --width 400 --alpha 1.5 --too-dark 58
# ============================================================

import argparse                # 读命令行参数（Day12 学过）
from pathlib import Path       # 造"地址牌"
import cv2                     # OpenCV：读图、缩放、灰度、调光、存图、写字
import numpy as np             # numpy：今天用 np.hstack 把两张图横着拼

# ---------- 1. 定义 5 个参数，并读回用户给的（或默认的）值 ----------
# argparse.ArgumentParser：造一个"参数说明书"
parser = argparse.ArgumentParser(description="图像批处理：缩放/灰度/拉对比度/出对比图和体检表")

# 每个 add_argument 登记一个参数：
#   第一个字符串是"名字"；"--in-dir" 这种带横线的，读回来叫 args.in_dir（横线变下划线）
#   type 决定读回来的类型（默认是字符串！）；default 是用户没给时的值
parser.add_argument("--in-dir",   default="D:/AI/zcode files/vision-lab/practice/day31_assets/in",  help="原料文件夹")
parser.add_argument("--out-dir",  default="D:/AI/zcode files/vision-lab/practice/day31_assets/out", help="成品文件夹")
parser.add_argument("--width",    type=int,   default=800, help="成品统一宽度，单位像素")
parser.add_argument("--alpha",    type=float, default=1.2, help="对比度倍数")
parser.add_argument("--too-dark", type=int,   default=55,  help="平均亮度低于它判太暗")
args = parser.parse_args()     # 真正去读命令行的这一句；读完 args.width 这些就能用了

# 把参数值接到后面代码要用的名字上（后面全程用这几个变量）
IN_DIR    = Path(args.in_dir)    # 原料文件夹的地址牌
OUT_DIR   = Path(args.out_dir)   # 成品文件夹的地址牌
CMP_DIR   = OUT_DIR / "cmp"      # 对比图单独放一个子文件夹，别和成品混在一起
TARGET_W  = args.width           # 成品宽度
ALPHA     = args.alpha           # 对比度倍数
TOO_DARK  = args.too_dark        # 太暗门槛
TOO_BRIGHT = 200                 # 过曝门槛（今天不做成参数，固定 200）

# ---------- 2. 建好两个发货区 ----------
# mkdir 加 exist_ok=True：已经有就不报错；parents=True：上级缺了一起建
OUT_DIR.mkdir(parents=True, exist_ok=True)
CMP_DIR.mkdir(parents=True, exist_ok=True)

# 把原料文件夹里所有 jpg 装成名单，按名字排序（顺序固定，方便一行行核对）
files = sorted(IN_DIR.glob("*.jpg"))
print(f"输入 {len(files)} 张 | 宽度 {TARGET_W} | alpha {ALPHA} | 太暗门槛 {TOO_DARK}")
print()

# ---------- 3. 一张一张过 ----------
report = []      # 体检结论的袋子
dark = 0         # 太暗计数
bright = 0       # 过曝计数

for p in files:                        # p：当前这张图的地址牌
    img = cv2.imread(p)                # 读原图 (3648, 5472, 3)
    h = img.shape[0]                   # 高
    w = img.shape[1]                   # 宽

    new_h = int(h * TARGET_W / w)                       # 等比算新高度（像素必须是整数）
    small = cv2.resize(img, (TARGET_W, new_h))          # 缩到（宽, 高）
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)      # 转灰度
    mean_v = gray.mean()                                # 平均亮度：体检指标

    # 三档判定：哪个先成立停哪个
    if mean_v < TOO_DARK:
        tag = "太暗"
        dark += 1
    elif mean_v > TOO_BRIGHT:
        tag = "过曝"
        bright += 1
    else:
        tag = "正常"

    final = cv2.convertScaleAbs(gray, alpha=ALPHA, beta=0)   # 拉对比度 = 成品
    ok = cv2.imwrite(OUT_DIR / p.name, final)                # 存成品；只换文件夹不换名

    # 拼对比图：左边彩色小图，右边成品（灰度图要先复制成三层才能拼）
    back = cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)
    pair = np.hstack((small, back))
    cv2.putText(pair, "原图", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
    cv2.putText(pair, "处理后", (TARGET_W + 10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
    cmp_ok = cv2.imwrite(CMP_DIR / f"cmp_{p.name}", pair)   # 对比图文件名前面加 cmp_

    line = f"{p.name}  {w}x{h} -> {TARGET_W}x{new_h}  亮度 {round(mean_v, 2)}  {tag}  存盘 {ok}  对比图 {cmp_ok}"
    print(line)
    report.append(line)

# ---------- 4. 汇总 + 写体检报告 ----------
print()
print(f"共处理 {len(report)} 张：太暗 {dark}，过曝 {bright}，正常 {len(report) - dark - bright}")

report_path = OUT_DIR / "体检报告.txt"
with open(report_path, "w", encoding="utf-8") as f:
    for line in report:
        f.write(line + "\n")
print("体检报告 ->", report_path)
print("对比图   ->", CMP_DIR)