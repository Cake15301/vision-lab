# ============================================================
# day38.py —— W06 第 6 天：轮廓（二）
#   boundingRect / minAreaRect / minEnclosingCircle / 长宽比 / 占空比
# 跑法：D:/python/python.exe day38.py
# 干的事：
#   1) 正外接矩形：给零件套一个"不能转"的方框，拿回位置和宽高
#   2) 长宽比与占空比：两个最常用的"体检指标"
#   3) 把零件拧斜 25 度：看正外接矩形怎么"框不准"
#   4) 最小外接矩形：能跟着零件转的框，4 个角怎么拿到
#   5) 最小外接圆：圆心 + 半径
#   6) 真齿轮：三种框一起画上，做一次综合体检
# ============================================================

from pathlib import Path
import cv2
import numpy as np

OUT_DIR = Path("D:/AI/zcode files/vision-lab/practice/day38_assets/out")   # 今天的成品目录
OUT_DIR.mkdir(parents=True, exist_ok=True)                                  # 没有就建；已有不报错

def to_bgr(image):
    """小工具①：一层灰度补成三层，才能往上写彩色线（和 Day37 一样）。"""
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

def big(image, w=480):
    """小工具②：把小图放大到 w 宽再存，为了看得清；最近邻插值 = 一个格子变一个方块，边不糊。"""
    return cv2.resize(image, (w, int(image.shape[0] * w / image.shape[1])),
                      interpolation=cv2.INTER_NEAREST)

# ---------- 1. 正外接矩形：给三个零件各套一个方框 ----------
# 这段在干什么：重建 Day37 那张零件散布图，给筛出来的 3 个大件挨个套方框，打印每个框的位置和大小。
parts = np.full((60, 80), 0, np.uint8)   # parts：60 行 80 列的"视野"，全黑（0 = 黑，255 = 白）
parts[5:25, 8:36]  = 255                 # 件 1：第 5~24 行、第 8~35 列涂白 → 20 行 × 28 列
parts[30:46, 50:66] = 255                # 件 2：16 行 × 16 列
parts[35:45, 10:28] = 255                # 件 3：10 行 × 18 列
dust = [(2, 70), (6, 74), (55, 4), (57, 20), (8, 44), (52, 72)]   # 6 个灰尘点，每项是 (行, 列)
for r, col in dust:                      # 挨个取灰尘点：r 是行号，col 是列号
    parts[r:r + 2, col:col + 2] = 255    # 每个灰尘点涂成 2×2 的白块

p_cnts, _ = cv2.findContours(parts, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   # 找轮廓（Day37）
kept = [c for c in p_cnts if cv2.contourArea(c) >= 60]   # 按面积筛掉灰尘（Day37 段 5 的闸门）
print("按面积筛完，剩", len(kept), "个零件")

for i, c in enumerate(sorted(kept, key=cv2.contourArea, reverse=True)):   # 按面积从大到小，挨个量。key是排序的依据：按照轮廓面积排序
    x, y, w, h = cv2.boundingRect(c)     # 一次接住 4 个数：左上角 (x, y) + 宽 w + 高 h
    print("  件", i + 1, "：左上角 (", x, ",", y, ") 宽", w, "高", h)

# ---------- 2. 长宽比与占空比 ----------
# 这段在干什么：给 3 个零件算"长宽比"和"占空比"两个体检指标，并打印。
# 结果预告：实心方块的占空比接近 1 但到不了 1——为什么，往下看。
for i, c in enumerate(sorted(kept, key=cv2.contourArea, reverse=True)):
    x, y, w, h = cv2.boundingRect(c)     # 框的 4 个数（段 1 讲过）
    area = cv2.contourArea(c)            # 轮廓面积：连点成线量出来的（昨天段 4，注意不是数格子）
    aspect = w / h                       # 长宽比 = 宽 ÷ 高；Python 的 / 会算出小数（比如 28/20 = 1.4）
    extent = area / (w * h)              # 占空比 = 轮廓面积 ÷ 框的面积（w*h 是框占的格数）
    print("  件", i + 1, "：w/h =", round(aspect, 3), " 占空比 =", round(extent, 4))

# ---------- 3. 把零件拧斜 25 度 ----------
# 这段在干什么：造一块 20 行×28 列的白块，绕图中心转 25 度，再量正外接矩形——看它变大多少。
slant = np.full((80, 120), 0, np.uint8)   # slant：80 行 120 列的新视野，全黑（比之前大，留出旋转空间）
slant[30:50, 46:74] = 255                 # 中间放一块白：第 30~49 行、第 46~73 列 → 20 行 × 28 列

M = cv2.getRotationMatrix2D((60, 40), 25, 1.0)    # Day28：造旋转矩阵——绕中心 (60, 40) 转 25 度、不缩放
# ⚠️ getRotationMatrix2D 的中心参数是 (x, y)，x 在前！这块白正好在图中央，所以绕它自己转
slant_rot = cv2.warpAffine(slant, M, (120, 80),   # Day28：按矩阵执行旋转；新图尺寸是 (宽, 高) = (120, 80)
                           flags=cv2.INTER_NEAREST)   # 最近邻插值：转完还是纯黑纯白，不产生灰色过渡

s_cnts, _ = cv2.findContours(slant_rot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   # 找斜块的轮廓
print("旋转后找到轮廓", len(s_cnts), "个")
sc = s_cnts[0]                            # sc：斜零件的轮廓（只有 1 个，直接取第 0 个）
sx, sy, sw, sh = cv2.boundingRect(sc)     # 斜零件的正外接矩形（变量名前缀 s = slant，免得和段 1 的混）
print("旋转后的正外接矩形：宽", sw, "高", sh, "（旋转前 28 × 20）")
print("轮廓面积还是", int(cv2.contourArea(sc)))

# ---------- 4. 最小外接矩形：框会跟着零件转 ----------
# 这段在干什么：给斜零件套"能转的最小框"，把三层套娃拆开看，再算出 4 个角画到图上。
rect = cv2.minAreaRect(sc)               # 跟斜零件的轮廓要一个"能转的最小框"
print("minAreaRect 返回的原样：", rect)   # 先看看它的真面目：三层套娃

center, size, angle = rect               # 拆第一层：center=中心、size=尺寸、angle=角度（三份各就各位）
cx, cy = center                          # 拆第二层：center 本身又是 (x, y)，再拆一次
rw, rh = size                            # 拆第二层：size 本身又是 (宽, 高)，再拆一次
print("中心 (", round(cx, 2), ",", round(cy, 2), ")  尺寸", (round(rw, 2), round(rh, 2)), "  角度", round(angle, 2))

if rw < rh:                              # ⚠️ OpenCV 不保证 rw 是长边！自己比一下再排
    long_e, short_e = rh, rw             #   rw 小 → rh 才是长边
else:
    long_e, short_e = rw, rh             #   否则 rw 就是长边
print("长边", round(long_e, 2), "短边", round(short_e, 2), "长宽比", round(long_e / short_e, 3))

sbox = cv2.boxPoints(rect)               # 问 rect 要它的 4 个角坐标 → 一个 4 行 2 列的小数表格
print("boxPoints 的形状", sbox.shape, "类型", sbox.dtype)
print(sbox)

sbox_int = sbox.astype(int)              # 画图必须整数坐标：astype(int) 把小数砍掉（Day37 用过）

vb = to_bgr(slant_rot)                   # 斜零件图补成彩色
cv2.rectangle(vb, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 1)   # 红：正外接矩形（不能转，框大一圈）
cv2.drawContours(vb, [sc], 0, (0, 0, 255), 1)                     # 红：零件自己的轮廓线
cv2.drawContours(vb, [sbox_int], 0, (0, 255, 0), 1)               # 绿：最小外接矩形（贴着零件转）
cv2.imwrite(str(OUT_DIR / "10_slant_minrect.png"), big(vb))       # 放大存图
print("存了 10_slant_minrect.png")

# ---------- 5. 最小外接圆 ----------
# 这段在干什么：造一个"圆形零件"，量它的正外接矩形和最小外接圆，算占空比，再画出来。
# cv2.circle 一人分两角：先当"造圆工具"画出圆形零件，最后再当"画圆工具"画外接圆。
ball = np.full((80, 80), 0, np.uint8)    # ball：80 行 80 列的黑图，放一个"圆形零件"
cv2.circle(ball, (40, 40), 20,           # 画圆：圆心 (x=40, y=40)、半径 20 像素（注意圆心是 x 在前！）
           255, -1)                      # 颜色 255（灰度图上就是白）；⚠️ 线宽 -1 = 涂满整个圆的内部
# ⚠️ 两个 -1 含义不同：drawContours 的 -1 是"列表里全画"，circle 的 -1 是"涂满"——别混

b_cnts, _ = cv2.findContours(ball, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   # 找圆的轮廓
bc = b_cnts[0]                           # bc：圆的轮廓（变量名前缀 b = ball）
bx, by, bw, bh = cv2.boundingRect(bc)    # 圆的正外接矩形（顺便复习段 1）
b_area = cv2.contourArea(bc)             # 圆的轮廓面积
print("圆的正外接矩形：", bw, "×", bh, " 轮廓面积", int(b_area))
print("圆的占空比 =", round(b_area / (bw * bh), 4))

ball_center, ball_r = cv2.minEnclosingCircle(bc)   # 拿回两样：圆心和半径，全是小数
print("minEnclosingCircle 拿回：圆心", ball_center, "半径", ball_r)
bcx, bcy = ball_center                   # 圆心是一对 (x, y)，再拆一次（和段 4 拆 center 一样）
print("半径取整 =", int(ball_r), "（int() 只砍小数，不四舍五入；四舍五入才用 round）")

vc = to_bgr(ball)                        # 补成彩色
cv2.circle(vc, (int(bcx), int(bcy)), int(ball_r),   # 画圆：⚠️ 圆心、半径都要整数，所以要 int() 包一层
           (0, 0, 255), 1)               # 红色、线宽 1（这回不涂满，只画圆周这条线）
cv2.imwrite(str(OUT_DIR / "11_ball_circle.png"), big(vc))
print("存了 11_ball_circle.png")

# ---------- 6. 真齿轮：三种框一起画上，做综合体检 ----------
# 这段在干什么：齿轮图走一遍老流水线拿到本体轮廓，然后 boundingRect / minAreaRect / minEnclosingCircle
#              三件套全上：红框（不能转）、绿框（能转）、蓝圆，画在同一张图上存出去，再打印体检表。
SRC      = Path("D:/AI/zcode files/vision-lab/practice/day33_assets/in/gear.png")   # 你的齿轮图
TARGET_W = 1000                                                                     # 统一缩到 1000 宽，单位像素
img  = cv2.imread(str(SRC))
ih, iw = img.shape[:2]                       # 原图的高 ih、宽 iw（shape 是先行后列）
gray = cv2.cvtColor(cv2.resize(img, (TARGET_W, int(ih * TARGET_W / iw))), cv2.COLOR_BGR2GRAY)
th, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)   # Otsu 自动找阈值（Day29）
clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))     # 开运算擦渣（Day36）
g_cnts, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   # 找轮廓（Day37）
gc = g_cnts[0]                               # gc：齿轮本体的轮廓（最外层只有 1 个）

gx, gy, gw, gh = cv2.boundingRect(gc)        # 红：正外接矩形（4 个数一次接住）
g_rect = cv2.minAreaRect(gc)                 # 绿：最小外接矩形（三层套娃）
g_center, g_size, g_angle = g_rect           # 和段 4 一样的拆法：中心 / 尺寸 / 角度
g_rw, g_rh = g_size                          # 尺寸再拆成宽、高
g_box  = cv2.boxPoints(g_rect).astype(int)   # 绿框的 4 个角，转整数才能画
g_c, g_r = cv2.minEnclosingCircle(gc)        # 蓝：最小外接圆（圆心 + 半径，全小数）
gcx, gcy = g_c                               # 圆心再拆一次

vd = to_bgr(gray)                            # 灰度齿轮补成彩色，往上面画
cv2.rectangle(vd, (gx, gy), (gx + gw, gy + gh), (0, 0, 255), 3)   # 红框：四边横平竖直
cv2.drawContours(vd, [g_box], 0, (0, 255, 0), 3)                  # 绿框：跟着齿轮的方向转
cv2.circle(vd, (int(gcx), int(gcy)), int(g_r), (255, 0, 0), 3)    # 蓝圆：最小外接圆（线宽 3，只画圆周）
cv2.imwrite(str(OUT_DIR / "12_gear_all.png"), vd)

print("齿轮综合体检：")
print("  红框（不能转）：", gw, "×", gh)
print("  绿框（能转）  ：", round(g_rw, 1), "×", round(g_rh, 1), " 角度", round(g_angle, 2))
print("  蓝圆（外接圆）：圆心 (", round(gcx, 1), ",", round(gcy, 1), ") 半径", round(g_r, 1))
print("  齿轮占空比 =", round(cv2.contourArea(gc) / (gw * gh), 4))

# ---------- 6（续）. 把段 1 答应的零件散布图红框版也存出来 ----------
va = to_bgr(parts)                           # 零件散布图补彩色（parts 是段 1 造的那张）
for c in kept:                               # 三个大件挨个画红框
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(va, (x, y), (x + w, y + h), (0, 0, 255), 1)   # 左上角 + 右下角=(x+w, y+h)，段 1 讲过
cv2.imwrite(str(OUT_DIR / "09_parts_bbox.png"), big(va))
print("存了 09_parts_bbox.png 和 12_gear_all.png")