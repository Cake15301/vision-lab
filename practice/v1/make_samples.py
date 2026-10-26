# ============================================================
# make_samples.py —— 造三件"样件"给 v1 练手
# 干的事：把一张齿轮原图变成三张同规格的灰度图，放进 v1/assets/in/
#   01_std.png   标准件（合格样件）—— 后面 pipeline.py 的"图纸"就是从它量出来的
#   02_small.png 偏小件（整件缩到 90%，模拟加工出来尺寸不够）
#   03_notch.png 缺角件（右下轮缘咬掉一块，模拟崩边）
# 跑法（在 v1 文件夹里）：D:/python/python.exe make_samples.py
# ============================================================

from pathlib import Path   # Path：造"地址牌"的工具（Day15 学过）
import cv2                 # OpenCV：读图、缩放、转灰度、存图
import numpy as np         # numpy：这里只用来造一块黑画布

# SRC：原料 —— W06 用的那张齿轮原图（1820 × 1380，彩色）
SRC = Path("D:/AI/zcode files/vision-lab/practice/day33_assets/in/gear.png")
# OUT：三件样件存哪儿
OUT = Path("D:/AI/zcode files/vision-lab/practice/v1/assets/in")
OUT.mkdir(parents=True, exist_ok=True)   # 文件夹没有就建；已经有了也不报错

TARGET_W = 1000   # 统一宽度（像素）：和 W05、W06 所有实验同规格，数字才能逐日对账

# ---------- 造 1 号：标准件 ----------
img = cv2.imread(str(SRC))   # 读原图（路径要 str() 包一层；中文路径会读空，这里全是英文）
ih, iw = img.shape[:2]       # 原图的高 ih、宽 iw —— shape 的顺序是"先高后宽"

# 先把宽度统一缩到 1000（高度按比例跟着变），再压成一层灰度
# 整条流水线只在灰度上干活：白色零件和深色背景的亮度差足够大，三通道彩色是浪费内存
std = cv2.cvtColor(
    cv2.resize(img, (TARGET_W, int(ih * TARGET_W / iw))),
    cv2.COLOR_BGR2GRAY)

H, W = std.shape   # 标准件的高 H = 758、宽 W = 1000 —— 下面两件要贴进同样大的画布
cv2.imwrite(str(OUT / "01_std.png"), std)
print("01_std.png   标准件", std.shape)

# ---------- 造 2 号：偏小件 ----------
# cv2.resize 的第二种用法：第二个位置给 None（占位，表示"尺寸这边不填"），
# 改用 fx / fy 给缩放倍数 —— 0.9 就是宽、高都变成原来的九成。
s = cv2.resize(std, None, fx=0.9, fy=0.9)      # 1000 × 758  →  900 × 682

# 造一块和标准件同样大的黑画布（np.zeros = 全 0 = 全黑，uint8 是像素值的类型：0~255）
canvas = np.zeros((H, W), np.uint8)

# 算"贴到正中"的左上角坐标：// 是整除（小数直接砍掉，留下的整数正好当切片下标用）
y0 = (H - s.shape[0]) // 2      # 竖方向留白的一半 = (758 - 682) // 2 = 38，记为行偏移
x0 = (W - s.shape[1]) // 2      # 横方向留白的一半 = (1000 - 900) // 2 = 50，记为列偏移

# 把缩小的图"贴"进画布正中 —— 切片赋值（Day36/39 用过）：
# 左边界 : 右边界 是列（横向），上边界 : 下边界 是行（纵向），别写反，从canvas取出的区域，赋值给s，s会覆盖canvas
canvas[y0:y0 + s.shape[0], x0:x0 + s.shape[1]] = s

cv2.imwrite(str(OUT / "02_small.png"), canvas)
print("02_small.png 偏小件", s.shape, "贴在", (x0, y0))

# ---------- 造 3 号：缺角件 ----------
bad = std.copy()   # ⚠️ 一定要 .copy()。写成 bad = std 的话，改 bad 会把标准件一起改坏
# 在右下轮缘咬掉一块：行 540~639（100 行高）、列 480~619（140 列宽），填成背景灰 30
bad[540:640, 480:620] = 30
cv2.imwrite(str(OUT / "03_notch.png"), bad)
print("03_notch.png 缺角件 已生成")

