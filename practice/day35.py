# ============================================================
# day35.py —— W06 第 3 天：Canny 边缘检测（双阈值 + 细线）
# 跑法：D:/python/python.exe day35.py
# 干的事：
#   1) 第一次用 Canny，拿到一条条单像素细线
#   2) 用一段数字看清"厚边变细线"是怎么发生的（非极大值抑制）
#   3) 分别拧一拧低阈值和高阈值，看它们各管什么
#   4) 用一张自制图证明"弱边要挨着强边才留下"（滞后）
#   5) 证明 Canny 不会替你做模糊：噪声图必须先滤波
#   6) 存图 + 拼两张对比图
# ============================================================

from pathlib import Path
import cv2
import numpy as np

# ---------- 0. 读图（和 Day34 一模一样，复习） ----------
IN_DIR  = Path("D:/AI/zcode files/vision-lab/practice/day33_assets/in")      # 原料：齿轮图所在文件夹
OUT_DIR = Path("D:/AI/zcode files/vision-lab/practice/day35_assets/out")     # 今天的成品目录
OUT_DIR.mkdir(parents=True, exist_ok=True)        # 没有就建；已经有了不报错

SRC      = IN_DIR / "gear.png"                    # 齿轮工件原图
TARGET_W = 1000                                   # 统一缩到 1000 宽（单位：像素）
img  = cv2.imread(str(SRC))                       # 读进内存；路径写错会返回 None
h, w = img.shape[:2]                              # 取高、宽（复习 Day33：shape 三个数是高、宽、层数）
gray = cv2.cvtColor(cv2.resize(img, (TARGET_W, int(h * TARGET_W / w))),  # resize 顺序是"宽、高"，别跟 shape 搞反
                    cv2.COLOR_BGR2GRAY)           # 三层彩色压成一层灰度
print("灰度", gray.shape, "平均亮度", round(gray.mean(), 2))

# ---------- 1. 第一次 Canny ----------
edges = cv2.Canny(gray, 50, 150)                  # 低阈值 50，高阈值 150
n_canny = int((edges > 0).sum())                  # 边缘图里白像素的个数（>0 就是边）
print("Canny(50,150) 白像素", n_canny, "占全图", round(n_canny / gray.size * 100, 2), "%")
print("图里出现的像素值", np.unique(edges))         # 看它是不是真的只有 0 和 255
cv2.imwrite(str(OUT_DIR / "01_canny_50_150.png"), edges)   # 存图（Canny 输出本身就是 0/255，不用转换）

# 昨天的老办法：Sobel X、Y 合成总强度，再看"大于 50"的算不算边
gx  = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # 左右方向变化（带正负号，必须浮点接）
gy  = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # 上下方向变化
mag = cv2.convertScaleAbs(cv2.magnitude(gx, gy))  # 勾股合成 + 取绝对值压回 0~255
n_mag = int((mag > 50).sum())                     # 昨天那种"粗边"的像素个数
print("昨天的 mag>50  白像素", n_mag, "占全图", round(n_mag / gray.size * 100, 2), "%")

# ---------- 2. 非极大值抑制：厚边怎么变成一条细线 ----------
print()
print("第400行 x=213..218  mag  :", mag[400, 213:219])    # 第 400 行、第 213 到 218 列，取 6 个数
print("第400行 x=213..218  canny:", edges[400, 213:219])  # 同一个位置，Canny 的结果
print("第400行 mag>50 的像素数", int((mag[400] > 50).sum()), " canny 白像素数", int((edges[400] > 0).sum()))

# ---------- 3. 两个阈值各管什么 ----------
print()
print("-- 低阈值固定 30，只抬高阈值 --")
for hi in [60, 90, 150]:                          # 三次实验：高阈值分别取 60、90、150
    print("   高", hi, "白像素", int((cv2.Canny(gray, 30, hi) > 0).sum()))
print("-- 高阈值固定 150，只动低阈值 --")
for lo in [30, 50, 100]:                          # 三次实验：低阈值分别取 30、50、100
    print("   低", lo, "白像素", int((cv2.Canny(gray, lo, 150) > 0).sum()))

# ---------- 4. 滞后：弱边要挨着强边才留下 ----------
# 先验证一次"整块改写"：造一张 4 行 6 列的空表，把中间 2 行 3 列改成 9
t = np.zeros((4, 6), np.uint8)                    # 4 行 6 列，全是 0 的空表格
t[1:3, 2:5] = 9                                   # 逗号左边管行（1、2），右边管列（2、3、4）
print()
print("整块改写后的表格：")
print(t)

# A 图：左边一块很亮的方块（强边），右边接一条很淡的横条（弱边）
syn_a = np.full((200, 340), 40, np.uint8)         # 200 行 340 列，先全部填 40（暗背景）；类型必须是 uint8
syn_a[20:180, 20:140] = 255                       # 第 20~179 行、第 20~139 列改成 255（强块）
syn_a[98:103, 140:340] = 62                       # 贴一条 5 行高的淡条，从强块右边沿一直到图右边

# B 图：一模一样，只是淡条整体往右挪 20 像素，跟强块之间空出一条缝
syn_b = np.full((200, 340), 40, np.uint8)
syn_b[20:180, 20:140] = 255
syn_b[98:103, 160:340] = 62                       # 淡条从第 160 列开始，140~159 列是背景（缝宽 20 像素）

ea = cv2.Canny(syn_a, 30, 150)                    # A 图用低 30、高 150
eb = cv2.Canny(syn_b, 30, 150)                    # B 图用同一组阈值
print("A 淡条接着强块：总白", int((ea > 0).sum()), " 淡条区(x>=160)白", int((ea[:, 160:] > 0).sum()))
print("B 淡条断开  ：总白", int((eb > 0).sum()), " 淡条区(x>=160)白", int((eb[:, 160:] > 0).sum()))

# 把这两张自制图和它们的 Canny 结果存下来，拼成一张四格的对比图
def to_bgr(image):
    """一层灰度补成三层，好写彩色字（数值和颜色都不变）。"""
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

def put_label(image, text):
    """在左上角写一行红字，返回写好字的图（复制一份再写，别弄脏原图）。"""
    out = image.copy()
    cv2.putText(out, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    return out

cv2.imwrite(str(OUT_DIR / "02_hyst_strong.png"), syn_a)   # A 原图：淡条接着强块
cv2.imwrite(str(OUT_DIR / "03_hyst_gap.png"), syn_b)      # B 原图：淡条断开
hyst  = np.hstack([put_label(to_bgr(syn_a), "A strong + weak attached"),   # 上排左：A 原图
                   put_label(to_bgr(syn_b), "B strong + weak gap")])       # 上排右：B 原图
hyst2 = np.hstack([put_label(to_bgr(ea), "A canny 30/150"),                # 下排左：A 的结果
                   put_label(to_bgr(eb), "B canny 30/150")])               # 下排右：B 的结果
hyst_all = np.vstack([hyst, hyst2])               # 上面一排原图、下面一排 Canny 结果
print("滞后对比图尺寸", hyst_all.shape)
cv2.imwrite(str(OUT_DIR / "04_hyst_canny.png"), hyst_all)

# ---------- 5. Canny 不替你做模糊 ----------
noisy_png = Path("D:/AI/zcode files/vision-lab/practice/day34_assets/out/07_noisy.png")  # 昨天存的高斯噪声图
noisy = cv2.cvtColor(cv2.imread(str(noisy_png)), cv2.COLOR_BGR2GRAY)   # 读回来再转灰度（存的是灰度，但 imread 默认读成三通道）
e_noisy = cv2.Canny(noisy, 50, 150)                                             # 不滤波直接求边
e_gauss = cv2.Canny(cv2.GaussianBlur(noisy, (5, 5), 0), 50, 150)                # 先高斯 5
e_bilat = cv2.Canny(cv2.bilateralFilter(noisy, 9, 75, 75), 50, 150)             # 先双边 9
e_med   = cv2.Canny(cv2.medianBlur(noisy, 5), 50, 150)                          # 先中值 5
print()
print("== 同一张噪声图，Canny(50,150) ==")
print("直接 Canny   ", int((e_noisy > 0).sum()), round((e_noisy > 0).sum() / noisy.size * 100, 2), "%")
print("高斯5 之后   ", int((e_gauss > 0).sum()), round((e_gauss > 0).sum() / noisy.size * 100, 2), "%")
print("双边9 之后   ", int((e_bilat > 0).sum()), round((e_bilat > 0).sum() / noisy.size * 100, 2), "%")
print("中值5 之后   ", int((e_med > 0).sum()), round((e_med > 0).sum() / noisy.size * 100, 2), "%")
print("干净图参照   ", n_canny, round(n_canny / gray.size * 100, 2), "%")

# 把三种滤波后的 Canny 结果存下来（中值那张只做数字对比，不存图）
cv2.imwrite(str(OUT_DIR / "05_canny_noisy.png"), e_noisy)       # 不滤波，满屏雪花
cv2.imwrite(str(OUT_DIR / "06_canny_gauss5.png"), e_gauss)      # 先高斯 5
cv2.imwrite(str(OUT_DIR / "07_canny_bilateral9.png"), e_bilat)  # 先双边 9

# ---------- 6. 再存三张 + 拼九格对比图 ----------
e_lo = cv2.Canny(gray, 30, 150)                   # 放宽低阈值（更容易把弱边接上）
e_hi = cv2.Canny(gray, 50, 200)                   # 收紧高阈值（只留最硬的边）
mag_thr = (mag > 50).astype(np.uint8) * 255       # 昨天那种"粗边"：真/假先转成 0/1，再乘 255 让1变成255，变成黑/白

cv2.imwrite(str(OUT_DIR / "08_mag_thr50.png"), mag_thr)      # 昨天的粗边（二值化后）
cv2.imwrite(str(OUT_DIR / "09_canny_30_150.png"), e_lo)      # 放宽低阈值的结果
cv2.imwrite(str(OUT_DIR / "10_canny_50_200.png"), e_hi)      # 收紧高阈值的结果

# to_bgr 和 put_label 在段 4 已经定义过，这里直接用，不用重写
row1 = np.hstack([put_label(to_bgr(gray),    "gray"),
                  put_label(to_bgr(mag_thr), "mag>50"),
                  put_label(to_bgr(edges),   "canny 50/150")])
row2 = np.hstack([put_label(to_bgr(e_lo),  "canny low=30"),
                  put_label(to_bgr(e_hi),  "canny high=200"),
                  put_label(to_bgr(noisy), "noisy")])
row3 = np.hstack([put_label(to_bgr(e_noisy), "canny on noisy"),
                  put_label(to_bgr(e_gauss), "canny + gauss5"),
                  put_label(to_bgr(e_bilat), "canny + bilateral9")])
grid = np.vstack([row1, row2, row3])              # 三行往下摞成九格
grid = cv2.resize(grid, (1500, int(grid.shape[0] * 1500 / grid.shape[1])))  # 总宽缩到 1500
print()
print("对比图尺寸", grid.shape)
cv2.imwrite(str(OUT_DIR / "cmp_canny.png"), grid)

print("结果图 ->", OUT_DIR)