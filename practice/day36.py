# ============================================================
# day36.py —— W06 第 4 天：形态学（腐蚀 / 膨胀 / 开运算 / 闭运算）
# 跑法：D:/python/python.exe day36.py
# 干的事：
#   1) 造一张"看得懂的小棋盘"，看清四种操作各改了什么
#   2) 用"两块 + 断口"的小棋盘，摸清闭运算能接多宽的断口
#   3) 回到真实齿轮图：把阈值面具弄脏，再擦干净
#   4) 存图 + 拼三张对比图
# ============================================================

from pathlib import Path
import cv2
import numpy as np

# ---------- 0. 读图（和 Day34／Day35 一样，复习） ----------
IN_DIR  = Path("D:/AI/zcode files/vision-lab/practice/day33_assets/in")      # 原料：齿轮图
OUT_DIR = Path("D:/AI/zcode files/vision-lab/practice/day36_assets/out")     # 今天的成品目录
OUT_DIR.mkdir(parents=True, exist_ok=True)        # 没有就建；已经有了不报错

SRC      = IN_DIR / "gear.png"                    # 齿轮工件原图
TARGET_W = 1000                                   # 统一缩到 1000 宽，单位像素
img  = cv2.imread(str(SRC))
h, w = img.shape[:2]
gray = cv2.cvtColor(cv2.resize(img, (TARGET_W, int(h * TARGET_W / w))),      # resize 顺序是"宽、高"
                    cv2.COLOR_BGR2GRAY)
print("灰度", gray.shape)

# ---------- 1. 造一张小棋盘：一块白方块 + 一个孤零零的白点 ----------
toy = np.full((10, 14), 0, np.uint8)     # 10 行 14 列，全黑（0 = 黑，255 = 白）
toy[2:8, 2:12] = 255                     # 把第 2~7 行、第 2~11 列整块涂白（6 行 × 10 列）
toy[9, 12] = 255                         # 在白块右下方点一个孤零零的白点（模拟噪点）
print()
print("小棋盘（1 = 白，0 = 黑）")
print((toy > 0).astype(int))             # (toy > 0) 是真/假表，astype(int) 变 0/1，更好读
print("白像素", int((toy > 0).sum()))     # 数一数有几个白格子

# ---------- 2. 结构元素（核）+ 腐蚀 ----------
k3 = np.ones((3, 3), np.uint8)           # 3×3 的"印章"：9 个格子全填 1
er = cv2.erode(toy, k3, iterations=1)    # 腐蚀：印章整体落在白区里，中心才保留白色， 1次腐蚀
print()
print("== 腐蚀 3×3 ==")
print("白像素", int((toy > 0).sum()), "->", int((er > 0).sum()))
print((er > 0).astype(int))

# ---------- 3. 膨胀 ----------
di = cv2.dilate(toy, k3, iterations=1)   # 膨胀：印章压到过的地方全变白
print()
print("== 膨胀 3×3 ==")
print("白像素", int((toy > 0).sum()), "->", int((di > 0).sum()))
print((di > 0).astype(int))

# ---------- 4. 开运算 = 先腐蚀再膨胀（擦噪点） ----------
op   = cv2.morphologyEx(toy, cv2.MORPH_OPEN, k3)   # 开运算：先腐蚀、再膨胀
op2  = cv2.dilate(er, k3)                          # 手动做一遍同样的两步，用来对账
print()
print("== 开运算 3×3 ==")
print("白像素", int((toy > 0).sum()), "->", int((op > 0).sum()))
print((op > 0).astype(int))
print("开运算 和 手动先腐蚀后膨胀 的最大差", int(np.abs(op.astype(int) - op2.astype(int)).max()))

# ---------- 5. 闭运算：两块之间隔多远还能接上（补黑洞，接断口） ----------
def two_blocks(gap):
    """造一张 13 行 22 列的小图：左右各一块白，中间空出 gap 列。"""
    g = np.full((13, 22), 0, np.uint8)          # 13 行 22 列，全黑
    g[3:10, 3:9] = 255                          # 左边那块：7 行 × 6 列
    g[3:10, 9 + gap:15 + gap] = 255             # 右边那块：和左边隔开 gap 列
    return g

g2 = two_blocks(2)                              # 断口 2 列
g4 = two_blocks(4)                              # 断口 4 列
k5 = np.ones((5, 5), np.uint8)                  # 5×5 的大印章

c2_3 = cv2.morphologyEx(g2, cv2.MORPH_CLOSE, k3)   # 2 格断口 + 3×3 印章
c4_3 = cv2.morphologyEx(g4, cv2.MORPH_CLOSE, k3)   # 4 格断口 + 3×3 印章
c4_5 = cv2.morphologyEx(g4, cv2.MORPH_CLOSE, k5)   # 4 格断口 + 5×5 印章

print()
print("== 闭运算 3×3：断口 2 格 ==")
print("白像素", int((g2 > 0).sum()), "->", int((c2_3 > 0).sum()))
print((c2_3 > 0).astype(int))
print()
print("== 闭运算 3×3：断口 4 格 ==")
print("白像素", int((g4 > 0).sum()), "->", int((c4_3 > 0).sum()))
print((c4_3 > 0).astype(int))
print()
print("== 闭运算 5×5：断口 4 格 ==")
print("白像素", int((g4 > 0).sum()), "->", int((c4_5 > 0).sum()))
print((c4_5 > 0).astype(int))
# 印章的边长，必须比断口宽。 能接上的最大断口 = 印章边长 − 1

# ---------- 6. 真实图：阈值面具 + 椒盐噪声 + 开运算 ----------
th, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)   # Otsu 自动找阈值，mask 是gray的二值化结果
print()
print("Otsu 阈值", th, "干净面具白像素", int((mask > 0).sum()))

rng    = np.random.default_rng(42)               # 随机数机器，种子 42（和 Day33 一样）
roll   = rng.random(gray.shape)                  # 每个像素抽一个 0~1 的小数
salt   = roll < 0.01                             # 约 1% 的像素 → 盐点（白）
pepper = roll > 0.99                             # 另约 1% → 椒点（黑）
m_sp = mask.copy()                               # 复制一份，别把干净面具弄脏
m_sp[salt]   = 255                               # 盐点涂白
m_sp[pepper] = 0                                 # 椒点涂黑
print("撒了盐点", int(salt.sum()), "椒点", int(pepper.sum()))    
print("弄脏后白像素", int((m_sp > 0).sum()), "和干净面具不一样", int((m_sp != mask).sum()))

for k in (3, 5, 7):
    kk = np.ones((k, k), np.uint8)
    cleaned = cv2.morphologyEx(m_sp, cv2.MORPH_OPEN, kk)
    print("开运算", k, "x", k, ": 白像素", int((cleaned > 0).sum()),
          "| 和干净面具差", int((cleaned != mask).sum()),
          "| 比干净面具少", int((mask > 0).sum()) - int((cleaned > 0).sum()))

# 分开做两个小实验：只撒盐 / 只撒椒，看开运算和闭运算各管哪一种
m_salt = mask.copy(); m_salt[salt] = 255          # 只有盐点（白点）
m_pep  = mask.copy(); m_pep[pepper] = 0           # 只有椒点（黑点）
print("只撒盐：开运算后差", int((cv2.morphologyEx(m_salt, cv2.MORPH_OPEN, k3) != mask).sum()),
      "| 闭运算后差", int((cv2.morphologyEx(m_salt, cv2.MORPH_CLOSE, k3) != mask).sum()))
print("只撒椒：开运算后差", int((cv2.morphologyEx(m_pep, cv2.MORPH_OPEN, k3) != mask).sum()),
      "| 闭运算后差", int((cv2.morphologyEx(m_pep, cv2.MORPH_CLOSE, k3) != mask).sum()))

c3   = cv2.morphologyEx(m_sp, cv2.MORPH_CLOSE, k3)               # 只做闭运算
oc3  = cv2.morphologyEx(c3, cv2.MORPH_OPEN, k3)                  # 先闭后开，别搞反！先接断口，再擦噪点
print("闭运算 3 x 3     和干净面具差", int((c3 != mask).sum()))
print("先闭后开 3 x 3   和干净面具差", int((oc3 != mask).sum()))

# ---------- 7. 存图 + 拼对比图 ----------
def to_bgr(image):
    """一层灰度补成三层，好写彩色字。"""
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

def put_label(image, text):
    """左上角写一行红字。"""
    out = image.copy()
    cv2.putText(out, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    return out

def big(image, w=280):
    """把小图放大到 w 宽，为了看得清；最近邻 = 一个格子变一个小方块，边不糊。"""
    return cv2.resize(image, (w, int(image.shape[0] * w / image.shape[1])),
                      interpolation=cv2.INTER_NEAREST)      # 放大图片时，最近邻直接抄离它最近的那个原格子的颜色

cv2.imwrite(str(OUT_DIR / "01_toy.png"),        big(toy))
cv2.imwrite(str(OUT_DIR / "02_toy_erode.png"),  big(er))
cv2.imwrite(str(OUT_DIR / "03_toy_dilate.png"), big(di))
cv2.imwrite(str(OUT_DIR / "04_toy_open.png"),   big(op))
cv2.imwrite(str(OUT_DIR / "05_gap2.png"),       big(g2, 400))
cv2.imwrite(str(OUT_DIR / "06_gap2_close3.png"), big(c2_3, 400))
cv2.imwrite(str(OUT_DIR / "07_gap4.png"),       big(g4, 400))
cv2.imwrite(str(OUT_DIR / "08_gap4_close3.png"), big(c4_3, 400))
cv2.imwrite(str(OUT_DIR / "09_gap4_close5.png"), big(c4_5, 400))
cv2.imwrite(str(OUT_DIR / "10_mask.png"),       mask)
cv2.imwrite(str(OUT_DIR / "11_mask_noisy.png"), m_sp)
cv2.imwrite(str(OUT_DIR / "12_open3.png"),  cv2.morphologyEx(m_sp, cv2.MORPH_OPEN, k3))
cv2.imwrite(str(OUT_DIR / "13_open5.png"),  cv2.morphologyEx(m_sp, cv2.MORPH_OPEN, k5))
cv2.imwrite(str(OUT_DIR / "14_open7.png"),  cv2.morphologyEx(m_sp, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8)))
cv2.imwrite(str(OUT_DIR / "15_open_then_close.png"), oc3)

cmp_toy = np.hstack([put_label(to_bgr(big(x)), t) for x, t in
                     [(toy, "1 board"), (er, "erode3"), (di, "dilate3"), (op, "open3")]])
cv2.imwrite(str(OUT_DIR / "cmp_toy.png"), cmp_toy)

cmp_gap = np.hstack([put_label(to_bgr(big(x, 400)), t) for x, t in
                     [(g2, "gap2"), (c2_3, "close3"), (g4, "gap4"),
                      (c4_3, "close3"), (c4_5, "close5")]])
cmp_gap = cv2.resize(cmp_gap, (1500, int(cmp_gap.shape[0] * 1500 / cmp_gap.shape[1])))
cv2.imwrite(str(OUT_DIR / "cmp_gap.png"), cmp_gap)

def small(image):
    """真实图统一缩到 400 宽，方便拼版。"""
    return cv2.resize(image, (400, int(image.shape[0] * 400 / image.shape[1])))

row1 = np.hstack([put_label(to_bgr(small(x)), t) for x, t in
                  [(mask, "mask"), (m_sp, "noisy"), (cv2.morphologyEx(m_sp, cv2.MORPH_OPEN, k3), "open3")]])
row2 = np.hstack([put_label(to_bgr(small(x)), t) for x, t in
                  [(cv2.morphologyEx(m_sp, cv2.MORPH_OPEN, k5), "open5"),
                   (cv2.morphologyEx(m_sp, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8)), "open7"),
                   (oc3, "close then open")]])
cmp_real = np.vstack([row1, row2])
cv2.imwrite(str(OUT_DIR / "cmp_real.png"), cmp_real)
print()
print("对比图尺寸", cmp_toy.shape, cmp_gap.shape, cmp_real.shape)
print("结果图 ->", OUT_DIR)