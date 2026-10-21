# ============================================================
# day34.py —— W06 Day2：边缘检测（一）Sobel X/Y + Laplacian
# 干的事：
#   1) 先把"边缘到底是什么"用一小串数字跑一遍
#   2) 拿齿轮工件图，用 Sobel 分别求"左右方向的变化"和"上下方向的变化"
#   3) 把两个方向合成一张"总边缘图"，再用 Laplacian 求二阶变化
#   4) 用实验证明"求边之前必须先滤波"
# 用法：D:/python/python.exe day34.py
# ============================================================

from pathlib import Path
import cv2
import numpy as np

# ---------- 0. 先把"边缘是什么"跑一遍（只验证概念，不参与后面的结果） ----------
a = np.array([10, 10, 10, 80, 80, 80, 80])        # 一条横线上从左往右的亮度：前 3 个暗，后 4 个亮
print("整条亮度        ", a)
print("从第 2 个开始取 ", a[1:])                   # 剪掉第一个，剩下的整体前移一位
print("取到倒数第 2 个 ", a[:-1])                  # 剪掉最后一个
print("相邻两个相减    ", a[1:] - a[:-1])          # 后一个减前一个 = 一阶差分

u = np.array([10, 80], dtype=np.uint8)            # 故意用"照片的整数类型"来装
print("uint8 80-10 =", u[1] - u[0])
print("uint8 10-80 =", u[0] - u[1])               # 负数装不住，会回绕成一个很大的正数

# ---------- 1. 读图、缩小、灰度 ----------
IN_DIR  = Path("D:/AI/zcode files/vision-lab/practice/day33_assets/in")    # 原料复用 Day33 的齿轮图
OUT_DIR = Path("D:/AI/zcode files/vision-lab/practice/day34_assets/out")   # 今天的成品目录
OUT_DIR.mkdir(parents=True, exist_ok=True)        # 没有就建；已经有了不报错

SRC      = IN_DIR / "gear.png"                    # 齿轮工件原图，1820 像素宽
TARGET_W = 1000                                   # 统一缩到 1000 宽，单位像素

img  = cv2.imread(str(SRC))                       # 读进内存；失败会返回 None
print("原图", img.shape)                           # (高, 宽, 层数)
h, w = img.shape[:2]                              # 只要高和宽
gray = cv2.cvtColor(cv2.resize(img, (TARGET_W, int(h * TARGET_W / w))),  # resize 顺序是"宽、高"
                    cv2.COLOR_BGR2GRAY)           # 三层彩色压成一层灰度
print("灰度", gray.shape, "平均亮度", round(gray.mean(), 2))
cv2.imwrite(str(OUT_DIR / "01_gray.png"), gray)

# ---------- 2. Sobel X：亮度"左右方向"的变化 ----------
gx_f = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)      # 结果带正负号，必须用浮点接
print("gx 浮点 min", gx_f.min(), "max", gx_f.max())    # 看到负号，就说明"有正有负"
gx = cv2.convertScaleAbs(gx_f)                         # 取绝对值并压进 0~255，才能存图
print("gx 存图后 max", gx.max(), "饱和到 255 的像素", int((gx == 255).sum()))
print("gx 里大于 50 的像素", int((gx > 50).sum()))      # 大于 50 的，今天算作"边"
cv2.imwrite(str(OUT_DIR / "02_sobel_x.png"), gx)

# ---------- 3. Sobel Y：亮度"上下方向"的变化 ----------
gy_f = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)      # 第 3、4 个参数是 dx、dy
gy = cv2.convertScaleAbs(gy_f)
print("gy 浮点 min", gy_f.min(), "max", gy_f.max())
print("gy 里大于 50 的像素", int((gy > 50).sum()))
cv2.imwrite(str(OUT_DIR / "03_sobel_y.png"), gy)

# ---------- 4. 总强度：两个方向合成一张 ----------
mag_f = cv2.magnitude(gx_f, gy_f)                      # 每个像素算"斜着走多远"（勾股定理）
print("mag 浮点 max", round(mag_f.max(), 2), "平均", round(mag_f.mean(), 2))
mag = cv2.convertScaleAbs(mag_f)
cv2.imwrite(str(OUT_DIR / "04_magnitude.png"), mag)

# ---------- 5. Laplacian：看"亮度坡度本身有没有变" ----------
lap_f = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
print("lap 浮点 min", lap_f.min(), "max", lap_f.max())
lap = cv2.convertScaleAbs(lap_f)
print("lap 里大于 50 的像素", int((lap > 50).sum()))
cv2.imwrite(str(OUT_DIR / "05_laplacian.png"), lap)

# 自己造一张"中间一条亮带"的小图，专门看 Laplacian 的"一明一暗双线"
left  = np.full((120, 80), 40, np.uint8)      # 120 行 80 列，全部填 40（暗）
mid   = np.full((120, 40), 200, np.uint8)     # 中间的亮带，宽 40 列
right = np.full((120, 80), 40, np.uint8)      # 右边再一块暗的
syn = np.hstack([left, mid, right])           # 左中右摞成 120×200 的合成图
syn_gx  = cv2.Sobel(syn, cv2.CV_64F, 1, 0, ksize=3)
syn_lap = cv2.Laplacian(syn, cv2.CV_64F, ksize=3)
print("合成图 sobelX     第60行 x=76..84: ", syn_gx[60, 76:85].astype(int))
print("合成图 laplacian  第60行 x=76..84: ", syn_lap[60, 76:85].astype(int))
print("合成图 laplacian  第60行 x=116..124:", syn_lap[60, 116:125].astype(int))

# ---------- 6. 求边之前必须滤波：加噪 / 滤波 再对比 ----------
rng = np.random.default_rng(7)                    # 和 Day33 同一台随机数机器、同一个种子
gn  = np.clip(gray.astype(np.float64) + rng.normal(0, 25, gray.shape), 0, 255).astype(np.uint8)
cv2.imwrite(str(OUT_DIR / "07_noisy.png"), gn)

def edge_x(image, thr):
    """求 Sobel X 边缘图；返回两样东西：大于 thr 的像素个数、边缘图本身。"""
    e = cv2.convertScaleAbs(cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3))
    return int((e > thr).sum()), e

n_clean, e_clean = edge_x(gray, 50)                                          # 干净图
n_noisy, e_noisy = edge_x(gn, 50)                                            # 只加噪
n_gauss, e_gauss = edge_x(cv2.GaussianBlur(gn, (5, 5), 0), 50)               # 加噪 + 高斯5
n_bilat, e_bilat = edge_x(cv2.bilateralFilter(gn, 9, 75, 75), 50)            # 加噪 + 双边9

print()
print("== Sobel X，大于 50 的像素，占全图比例 ==")
print("干净图  ", n_clean, round(n_clean / gray.size * 100, 2), "%")
print("加噪声  ", n_noisy, round(n_noisy / gray.size * 100, 2), "%")
print("高斯5后 ", n_gauss, round(n_gauss / gray.size * 100, 2), "%")
print("双边9后 ", n_bilat, round(n_bilat / gray.size * 100, 2), "%")

cv2.imwrite(str(OUT_DIR / "08_edge_noisy.png"), e_noisy)
cv2.imwrite(str(OUT_DIR / "09_edge_gauss5.png"), e_gauss)
cv2.imwrite(str(OUT_DIR / "10_edge_bilateral9.png"), e_bilat)

# ---------- 7. 拼九格对比图 ----------
def to_bgr(image):
    """一层灰度补成三层，好写彩色字（数值不变，颜色也不变）。"""
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

def put_label(image, text):
    """在左上角写一行红字，返回写好字的图。"""
    out = image.copy()   # 复制一份再写，别把原图弄脏
    cv2.putText(out, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    return out

row1 = np.hstack([put_label(to_bgr(gray), "gray"),
                  put_label(to_bgr(gx),   "sobel X"),
                  put_label(to_bgr(gy),   "sobel Y")])
row2 = np.hstack([put_label(to_bgr(mag),  "magnitude"),
                  put_label(to_bgr(lap),  "laplacian"),
                  put_label(to_bgr(gn),   "noisy")])
row3 = np.hstack([put_label(to_bgr(e_noisy), "edge on noisy"),
                  put_label(to_bgr(e_gauss), "edge + gauss5"),
                  put_label(to_bgr(e_bilat), "edge + bilateral9")])

grid = np.vstack([row1, row2, row3])              # 三行往下摞
grid = cv2.resize(grid, (1500, int(grid.shape[0] * 1500 / grid.shape[1])))
print()
print("对比图尺寸", grid.shape)
cv2.imwrite(str(OUT_DIR / "cmp_edges.png"), grid)

print("结果图 ->", OUT_DIR)