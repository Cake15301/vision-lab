# ============================================================
# day33.py —— W06 Day1：滤波去噪
# 干的事：拿齿轮工件图，故意加两种噪声（椒盐、高斯），
#        再用 4 种滤波分别去洗，比谁洗得干净
# 用法：D:/python/python.exe day33.py
# ============================================================

from pathlib import Path     # 造"地址牌"
import cv2                   # OpenCV：读图、缩放、灰度、滤波、存图、写字
import numpy as np           # numpy：造随机数、拼图、算平均差

IN_DIR  = Path("D:/AI/zcode files/vision-lab/practice/day33_assets/in")   # 原料文件夹的地址牌
OUT_DIR = Path("D:/AI/zcode files/vision-lab/practice/day33_assets/out")  # 成品文件夹的地址牌
OUT_DIR.mkdir(parents=True, exist_ok=True)   # 成品文件夹没有就建；已经有了不报错

SRC      = IN_DIR / "gear.png"   # 原料：你自己渲染的齿轮工件图（1820 宽）
TARGET_W = 1000                  # 统一宽度，单位像素（原图太宽，先缩到 1000）

# ---------- 1. 读图、缩小、灰度 ----------
img = cv2.imread(str(SRC))        # 读进内存；读失败会返回 None
print("原图", img.shape)           # (高, 宽, 层数) —— 原图是 (1380, 1820, 3)

h, w = img.shape[:2]              # 只要高和宽，不要层数
new_h = int(h * TARGET_W / w)     # 等比例算新高度；像素个数必须是整数，所以套 int()
small = cv2.resize(img, (TARGET_W, new_h))          # 注意 resize 的顺序是"宽、高"
gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)     # 三层彩色压成一层灰度
print("小图", small.shape, "灰度", gray.shape)
print("干净图平均亮度", round(gray.mean(), 2))       # 平均亮度：后面判断"洗得干不干净"的参照
cv2.imwrite(str(OUT_DIR / "01_clean.png"), gray)    # 存成第一张结果图

# ---------- 2. 造椒盐噪声（白点叫盐，黑点叫椒） ----------
rng  = np.random.default_rng(42)      # 随机数机器；42 是种子，同一个种子每次抽签结果一样
roll = rng.random(gray.shape)         # 给每个像素抽一个 0~1 的小数，形状和 gray 一样

salt   = roll < 0.02                  # 小于 0.02 的位置 → True（约 2% 的像素）
pepper = roll > 0.98                  # 大于 0.98 的位置 → True（另约 2%）

sp = gray.copy()                      # 复制一份再改；直接 sp = gray 会连干净图一起改
sp[salt]   = 255                      # 把 True 的位置一次性改成 255（白点）
sp[pepper] = 0                        # 把 True 的位置一次性改成 0（黑点）

print("盐点", int(salt.sum()), "椒点", int(pepper.sum()), "总像素", gray.size)
print("坏点占比", round((salt.sum() + pepper.sum()) / gray.size * 100, 2), "%")
cv2.imwrite(str(OUT_DIR / "02_saltpepper.png"), sp)

# ---------- 3. 造高斯噪声（每个像素都加一个"钟形"随机数） ----------
rng2  = np.random.default_rng(7)            # 另起一台机器、另用一个种子
noise = rng2.normal(0, 25, gray.shape)      # 平均 0、标准差 25 的小数，形状同 gray

gn = gray.astype(np.float64) + noise        # 先转成小数类型再相加，避免整数溢出
gn = np.clip(gn, 0, 255)                    # 超过 255 卡 255，小于 0 卡 0
gn = gn.astype(np.uint8)                    # 转回 8 位整数，图片才能存

print("高斯噪声图 min", gn.min(), "max", gn.max(), "平均亮度", round(gn.mean(), 2))
cv2.imwrite(str(OUT_DIR / "03_gauss.png"), gn)

# ---------- 4. 四种滤波器，各洗一遍 ----------
def four_filters(image):
    """拿一张图，返回 4 个滤波结果；每个结果贴一个英文标签，方便后面按名字取。"""
    return {
        "mean5":      cv2.blur(image, (5, 5)),            # 均值：5×5 窗口等权平均
        "gauss5":     cv2.GaussianBlur(image, (5, 5), 0), # 高斯：5×5 窗口钟形加权；0 = 自动算标准差
        "median5":    cv2.medianBlur(image, 5),           # 中值：5×5 窗口取中位数（参数是一个整数！）
        "bilateral9": cv2.bilateralFilter(image, 9, 75, 75),  # 双边：直径 9，颜色阈值 75，距离影响 75
    }

sp_results = four_filters(sp)    # 椒盐图的 4 张结果
gn_results = four_filters(gn)    # 高斯图的 4 张结果

for name in sp_results:                                   # 遍历字典 = 逐个拿"标签"
    cv2.imwrite(str(OUT_DIR / ("sp_" + name + ".png")), sp_results[name])
    print("已存 sp_" + name + ".png")

for name in gn_results:
    cv2.imwrite(str(OUT_DIR / ("gn_" + name + ".png")), gn_results[name])
    print("已存 gn_" + name + ".png")

# ---------- 5. 拼六格对比图 ----------
def to_bgr(image):
    """一层灰度补成三层，好写彩色字（数值不变，颜色也不变）。"""
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

def put_label(image, text):
    """在一张图左上角写一行红字，返回写好字的图。"""
    out = image.copy()   # 复制一份再写，别把原图弄脏
    cv2.putText(out, text, (10, 40),              # (10, 40) 是文字起点，左上往右 10、往下 40
                cv2.FONT_HERSHEY_SIMPLEX, 1.0,    # 字体、字号
                (0, 0, 255), 3)                   # 颜色 BGR=(0,0,255) 是红色；粗细 3
    return out

# 第一行：干净图、椒盐图、中值结果
row1 = np.hstack([put_label(to_bgr(gray), "clean"),
                  put_label(to_bgr(sp), "salt&pepper"),
                  put_label(to_bgr(sp_results["median5"]), "median5")])

# 第二行：均值、高斯、双边
row2 = np.hstack([put_label(to_bgr(sp_results["mean5"]), "mean5"),
                  put_label(to_bgr(sp_results["gauss5"]), "gauss5"),
                  put_label(to_bgr(sp_results["bilateral9"]), "bilateral9")])

sp_grid = np.vstack([row1, row2])       # 第二行摞在第一行下面
sp_grid = cv2.resize(sp_grid, (1500, int(sp_grid.shape[0] * 1500 / sp_grid.shape[1])))    # 缩放比例 = 目标宽 / 原宽，再用高乘以缩放比
print("对比图尺寸", sp_grid.shape)       # 3 张 1000 宽的图拼起来是 3000 宽，再缩到 1500
cv2.imwrite(str(OUT_DIR / "cmp_saltpepper.png"), sp_grid)

# ---------- 6. 打分：每个结果和干净图的平均差 ----------
def score(base, filtered):
    """逐像素求差、取绝对值、再求平均；越小 = 越接近干净图。"""
    return cv2.absdiff(base, filtered).mean()

print()
print("== 椒盐噪声：和干净图的平均差（越小越接近原图）==")
print("不滤波", round(score(gray, sp), 2))        # 先看不洗的分数，当参照
for name, result in sp_results.items():           # .items() 一次拿"标签 + 内容"两样
    print(name, round(score(gray, result), 2))

print()
print("== 高斯噪声：和干净图的平均差 ==")
print("不滤波", round(score(gray, gn), 2))
for name, result in gn_results.items():
    print(name, round(score(gray, result), 2))
    

# 裁剪用切片：先取行范围，再取列范围（Day27 学过裁剪）
y0, y1 = 250, 550    # 从上往下，第 250 行到第 550 行
x0, x1 = 200, 500    # 从左往右，第 200 列到第 500 列

crop_noise  = put_label(to_bgr(gn[ y0:y1, x0:x1 ]), "gauss noise")
crop_mean   = put_label(to_bgr(gn_results["mean5"][ y0:y1, x0:x1 ]), "mean5")
crop_bilat  = put_label(to_bgr(gn_results["bilateral9"][ y0:y1, x0:x1 ]), "bilateral9")
crop_clean  = put_label(to_bgr(gray[ y0:y1, x0:x1 ]), "clean")

zoom_top = np.hstack([crop_noise, crop_mean])
zoom_bot = np.hstack([crop_bilat, crop_clean])
cv2.imwrite(str(OUT_DIR / "cmp_zoom.png"), np.vstack([zoom_top, zoom_bot]))