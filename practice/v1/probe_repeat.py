# probe_repeat.py —— 同一件零件换个成像条件再量一次，看读数能稳到多少
# 干的事：造 4 个版本（原图 / 轻微失焦 / 工件退远 2% / 视野里挪 1 像素），逐个量，算偏差和余量
# 跑法（在 v1 文件夹里）：D:/python/python.exe probe_repeat.py
# 为什么需要它：公差不能凭感觉定 —— 先量出"同一件的读数会晃多少"，才谈得上给几倍余量

from pathlib import Path
import cv2          # 存图、模糊、缩放都靠它
import numpy as np  # 挪格子用 np.roll
import pipeline     # 有守门员，导入它不会自己跑一批

# SRC：被试零件的地址 —— 从"本脚本所在的文件夹"（.parent）出发，站在哪个文件夹跑都对
SRC = Path(__file__).parent / "assets" / "in" / "01_std.png"

# PROBE：证据图的存放地 —— 同样跟着脚本走
PROBE = Path(__file__).parent / "assets" / "probe"
PROBE.mkdir(parents=True, exist_ok=True)

img = cv2.imread(str(SRC))           # imread 不认 Path 对象，和 pipeline 里一样要 str()

# 四个版本。键就是文件名，所以只能用英文（OpenCV 存中文名会静默失败、回 False）
#   c1_orig    原图               —— 基准，应该正好是 186121.0 / 0.7373
#   c2_blur5   5×5 高斯模糊        —— 模拟轻微失焦 / 镜头有点脏
#   c3_small98 整体缩到 98%        —— 模拟工件退远 2%（同样的视野里，零件显得更小）
#   c4_right1  整张图右移 1 像素    —— 模拟工件在视野里挪了一格
cases = {
    "c1_orig":    img,
    "c2_blur5":   cv2.GaussianBlur(img, (5, 5), 0),          # (5,5)=模糊核大小；第 3 个 0＝让 OpenCV 自己算强度
    "c3_small98": cv2.resize(img, None, fx=0.98, fy=0.98),   # 第 2 个给 None＝宽高不用手写，按 fx/fy 比例算
    "c4_right1":  np.roll(img, 1, axis=1),                   # axis=1＝左右方向挪；挪出去的那列从另一边补回来
}

worst_a = 0.0   # 面积偏差里绝对值最大的（小数；0.0017 就是 0.17%）
worst_f = 0.0   # 占空比偏差里绝对值最大的
print("版本 | 面积 | 面积偏差 | 占空比 | 占空比偏差")

for name, im in cases.items():
    f = PROBE / (name + ".png")
    cv2.imwrite(str(f), im)             # 先存成文件：measure() 要的是一张图的地址，不是内存里的数组
    m = pipeline.measure(f)             # 借用流水线的 measure，不抄第二遍（抄一遍就会出现第二种答案）
    dev_a = (m["面积"] - pipeline.SPEC_AREA) / pipeline.SPEC_AREA   # 相对偏差，0.0017 = 0.17%
    dev_f = round(m["占空比"], 4) - pipeline.SPEC_FILL
    worst_a = max(worst_a, abs(dev_a))  # max(甲, 乙) 取大的那个；abs 取绝对值（负的也变正）
    worst_f = max(worst_f, abs(dev_f))
    print(name, "|", round(m["面积"], 1), "|", round(dev_a * 100, 4), "%",
          "|", round(m["占空比"], 4), "|", round(dev_f, 4))

print()
print("最大面积偏差", round(worst_a * 100, 4), "%  ->  ±5% 是它的", round(pipeline.TOL_AREA / worst_a), "倍")
print("最大占空比偏差", round(worst_f, 4), "  ->  ±0.02 是它的", round(pipeline.TOL_FILL / worst_f, 1), "倍")