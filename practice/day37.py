# ============================================================
# day37.py —— W06 第 5 天：轮廓（一）
#   findContours / 三种检索模式 / contourArea / arcLength / 按面积筛选
# 跑法：D:/python/python.exe day37.py
# 干的事：
#   1) 一块白方块：看清"轮廓"是什么、一个轮廓长什么样
#   2) "回"字图：看清 EXTERNAL / LIST / TREE 三种检索模式的区别
#   3) 量面积与周长：为什么"轮廓面积"和"数白格子"对不上
#   4) 零件散布图：按面积筛掉灰尘
#   5) 真实齿轮图：一整个工件到底算几个轮廓
#   6) 把轮廓画出来 + 存图 + 拼对比图
# ============================================================

from pathlib import Path
import cv2
import numpy as np

OUT_DIR = Path("D:/AI/zcode files/vision-lab/practice/day37_assets/out")   # 今天的成品目录
OUT_DIR.mkdir(parents=True, exist_ok=True)                                  # 没有就建；已有不报错

# ---------- 1. 第一张棋盘：一个白方块 → 几个轮廓？ ----------
board = np.full((6, 6), 0, np.uint8)   # 6 行 6 列全黑（0 = 黑，255 = 白）
board[1:5, 1:5] = 255                  # 第 1~4 行、第 1~4 列涂白 → 一块 4×4 的白方块
print("棋盘（1 = 白，0 = 黑）")
print((board > 0).astype(int))         # 转成 0/1 再打印，比看 255 清楚

# findContours 要找的是"白色的连通块"，只认 0 和 255 的黑白图
cnts, hier = cv2.findContours(board,                    # 输入：黑白图
                              cv2.RETR_EXTERNAL,        # 检索模式：只找最外层
                              cv2.CHAIN_APPROX_SIMPLE)  # 近似法：只记拐点，直线中间的点省掉
print()
print("找到轮廓", len(cnts), "个")
print("hierarchy（层级表）的形状", hier.shape)            # (1, 1, 4)：1 个轮廓，每个 4 个数

c0 = cnts[0]                            # 第一个轮廓
print("c0 是什么类型", type(c0), "形状", c0.shape)        # (4, 1, 2)：4 个点，每个点有 2 个数字: x、y
print("c0 原样打印：")
print(c0)
print("c0 压平后（一行一个点）：")
print(c0.reshape(-1, 2))                # reshape(-1, 2)：列数定死 2（即x、y），行数自动算
print("点数 len(c0) =", len(c0))

cnts_none, _ = cv2.findContours(board, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
print("换成 CHAIN_APPROX_NONE 后的点数", len(cnts_none[0]))   # 每个边界像素都记，点数变多

# ---------- 2. "回"字图：EXTERNAL 与 LIST 差在哪 ----------
ring = np.full((9, 9), 0, np.uint8)    # 9 行 9 列全黑
ring[1:8, 1:8] = 255                   # 第 1~7 行、第 1~7 列涂白（7×7 的白块）
ring[3:6, 3:6] = 0                     # 再把正中间 3×3 挖黑 → 变成"回"字
print()
print("回字图（1 = 白）")
print((ring > 0).astype(int))

e_cnts, _ = cv2.findContours(ring, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 只要最外层
l_cnts, _ = cv2.findContours(ring, cv2.RETR_LIST,     cv2.CHAIN_APPROX_SIMPLE)  # 全都要
print()
print("EXTERNAL 找到", len(e_cnts), "个")
print("LIST     找到", len(l_cnts), "个")
for i, c in enumerate(l_cnts):              # enumerate：边取轮廓边给编号
    print("  第", i, "个轮廓：点数", len(c), "点坐标", c.reshape(-1, 2).tolist())

# ---------- 3. TREE：把"谁套着谁"也记下来 ----------
t_cnts, t_hier = cv2.findContours(ring, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print()
print("TREE 找到", len(t_cnts), "个")
print("t_hier 形状", t_hier.shape)            # (1, 2, 4)：1 张表，2 个轮廓，每个 4 个数
print("四元组 = [下一个, 上一个, 第一个孩子, 父亲]，-1 表示没有")
print(t_hier[0])
print("第 0 个轮廓：", t_hier[0][0])          # 先取第一层，再从拿到的结果里取一层
print("第 1 个轮廓：", t_hier[0][1])

# ---------- 4. 面积与周长：为什么和"数格子"对不上 ----------
print()
print("白格子数（直接数像素）", int((board > 0).sum()))
print("轮廓面积 contourArea", cv2.contourArea(c0))
print("周长 arcLength(True)  闭合", cv2.arcLength(c0, True))
print("周长 arcLength(False) 不闭合", cv2.arcLength(c0, False))
print("回字两个轮廓各自的面积：")
for i, c in enumerate(l_cnts):
    print("  第", i, "个：面积", cv2.contourArea(c), "点数", len(c))

# ---------- 5. 零件散布图：按面积筛掉灰尘 ----------
parts = np.full((60, 80), 0, np.uint8)   # 60 行 80 列的"视野"，全黑
parts[5:25, 8:36]  = 255                 # 件 1：20 行 × 28 列
parts[30:46, 50:66] = 255                # 件 2：16 行 × 16 列
parts[35:45, 10:28] = 255                # 件 3：10 行 × 18 列
dust = [(2, 70), (6, 74), (55, 4), (57, 20), (8, 44), (52, 72)]   # 6 个灰尘点，每个坐标是 (行, 列)
for r, col in dust:                      # 每次从列表里取一组 (r, col)，r 是行号、col 是列号
    parts[r:r + 2, col:col + 2] = 255    # 每个灰尘点涂成 2×2

p_cnts, _ = cv2.findContours(parts, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print()
print("零件散布图：找到轮廓", len(p_cnts), "个")
print("每个轮廓的面积", [int(cv2.contourArea(c)) for c in p_cnts])   # 列表推导式：把每个面积收进列表

kept = []                                # 老实的写法：先准备空列表
for c in p_cnts:                         # 挨个看
    if cv2.contourArea(c) >= 60:         # 面积够大才算零件
        kept.append(c)                   # 收进来
print("按面积 >= 60 筛完，剩", len(kept), "个")
kept2 = [c for c in p_cnts if cv2.contourArea(c) >= 60]   # 同一件事的简写，结果必须一样
print("列表推导式写法，剩", len(kept2), "个")

for c in sorted(kept, key=cv2.contourArea, reverse=True):   # 按面积从大到小排
    print("  面积", int(cv2.contourArea(c)), "周长", cv2.arcLength(c, True))

area_sum = 0                            # 累计器：先归零
for c in kept:                          # 把留下的 3 个轮廓挨个过一遍
    area_sum = area_sum + cv2.contourArea(c)   # 每过一遍就往上加它自己的面积
print("白像素总数", int((parts > 0).sum()), "| 三个大件轮廓面积之和", int(area_sum))

# ---------- 6. 真实齿轮图：整个工件算几个轮廓 ----------
SRC      = Path("D:/AI/zcode files/vision-lab/practice/day33_assets/in/gear.png")   # 你的齿轮图
TARGET_W = 1000                                                                     # 统一缩到 1000 宽，单位像素
img  = cv2.imread(str(SRC))
h, w = img.shape[:2]
gray = cv2.cvtColor(cv2.resize(img, (TARGET_W, int(h * TARGET_W / w))), cv2.COLOR_BGR2GRAY)
th, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)   # Otsu 自动找阈值
clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))      # 开运算擦掉小碎块
print()
print("齿轮图：Otsu 阈值", th, "| 开运算后白像素", int((clean > 0).sum()))

g_cnts, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
gc = g_cnts[0]                           # 最外层只有 1 个，就是工件本体
print("EXTERNAL 找到", len(g_cnts), "个")
print("  面积", int(cv2.contourArea(gc)), "周长", round(cv2.arcLength(gc, True), 1), "点数", len(gc))

g_list, _ = cv2.findContours(clean, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
print("LIST 找到", len(g_list), "个")
print("  面积从大到小前 5", sorted([int(cv2.contourArea(c)) for c in g_list], reverse=True)[:5])

# ---------- 7. 把轮廓画出来 + 存图 + 拼对比图 ----------
def to_bgr(image):
    """一层灰度补成三层，才能往上写彩色线。"""
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

def put_label(image, text):
    """左上角写一行红字。"""
    out = image.copy()                                                  # 先复制，别改原图
    cv2.putText(out, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    return out

def big(image, w=360):
    """把小图放大到 w 宽，为了看得清；最近邻 = 一个格子变一个方块，边不糊。"""
    return cv2.resize(image, (w, int(image.shape[0] * w / image.shape[1])),
                      interpolation=cv2.INTER_NEAREST)

# 全部 9 个轮廓：红线画
canvas_all = to_bgr(parts)
cv2.drawContours(canvas_all, p_cnts, -1, (0, 0, 255), 1)   # -1 = 列表里每个都画；1 = 线宽 1 像素
# 筛完的 3 个轮廓：绿线画粗一点
canvas_kept = to_bgr(parts)
cv2.drawContours(canvas_kept, kept, -1, (0, 255, 0), 2)

# 齿轮：把所有 EXTERNAL 轮廓画在灰度图上
gear_view = to_bgr(gray)
cv2.drawContours(gear_view, g_cnts, -1, (0, 255, 0), 3)

b01 = put_label(big(to_bgr(board), 360), "board 6x6")
b02 = put_label(big(to_bgr(ring), 360), "ring 9x9")
b03 = put_label(big(canvas_all, 480), "all 9")
b04 = put_label(big(canvas_kept, 480), "kept 3")

cv2.imwrite(str(OUT_DIR / "01_board.png"), b01)
cv2.imwrite(str(OUT_DIR / "02_ring.png"), b02)
cv2.imwrite(str(OUT_DIR / "03_parts_all.png"), b03)
cv2.imwrite(str(OUT_DIR / "04_parts_kept.png"), b04)
cv2.imwrite(str(OUT_DIR / "05_gear_contour.png"), gear_view)
cv2.imwrite(str(OUT_DIR / "cmp_parts.png"), np.hstack([b03, b04]))              # 横着拼
cv2.imwrite(str(OUT_DIR / "cmp_gear.png"), np.hstack([to_bgr(gray), gear_view])) # 原图 vs 带轮廓

print()
print("存了 7 张图到", OUT_DIR)