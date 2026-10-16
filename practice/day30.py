import cv2
import numpy as np

GH  = "D:/AI/zcode files/vision-lab/practice/day26_assets/grace_hopper.jpg"   # 人像小图 600×512
OK1 = "D:/AI/zcode files/vision-lab/practice/day26_assets/ok1.jpg"            # 你的编码器样本

# ========== 段 0：亲手数一遍"亮度分布" ==========
gray = cv2.cvtColor(cv2.imread(GH), cv2.COLOR_BGR2GRAY)
print("人像灰度 shape =", gray.shape, " 总像素数 =", gray.size)
print("最暗 =", gray.min(), " 最亮 =", gray.max(),
      " 平均 =", round(gray.mean(), 2), " 标准差 =", round(gray.std(), 2))

bucket = gray // 32          # 每 32 个亮度算一格，0~255 一共 8 格
for k in range(8):
    lo = k * 32
    hi = lo + 31
    print(f"第 {k} 格（{lo}-{hi}）: {(bucket == k).sum()} 个像素")

# ========== 段 1：让 cv2 用官方机器数一遍 ==========
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
print("hist.shape =", hist.shape, " hist.dtype =", hist.dtype)
print("第 0 桶 =", hist[0], " 第 128 桶 =", hist[128], " 第 255 桶 =", hist[255])
print("最高的一根柱子 =", int(hist.max()), "它在第", hist.argmax(), "号桶")
print("所有桶加起来 =", int(hist.sum()), " 像素总数 =", gray.size)
print("直接数一下 灰度==14 的像素 =", (gray == 14).sum())       # 数像素的方法

# 跟段 0 的手工对账：第 k 格 = 第 32k 到 32k+31 号桶加起来
for k in range(3):
    print(f"第 {k} 格手数 = {(bucket == k).sum()}，"
          f"calcHist 对应的 32 个桶之和 = {int(hist[k*32:(k+1)*32].sum())}")

# ========== 段 2：把直方图画出来 ==========
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]   # 让图里的中文正常显示
plt.rcParams["axes.unicode_minus"] = False              # 让负号正常显示

plt.figure(figsize=(8, 3))     # 开一张 8×3 英寸的新画布
plt.plot(hist)                 # 把 256 个数当成 256 个点，连成线
plt.title("grace_hopper 灰度直方图")
plt.xlabel("亮度（0=黑，255=白）")
plt.ylabel("像素个数")
plt.tight_layout()
plt.savefig("day30_直方图_人像.png")
plt.close()                    # 把这张画布收掉
print("图已保存：day30_直方图_人像.png")

# ========== 段 3：两个旋钮 ==========
print("原图      平均", round(gray.mean(), 2), "最小", gray.min(), "最大", gray.max())

# 处理像素加减时，永远用 convertScaleAbs，不要直接 + 数字
bright = cv2.convertScaleAbs(gray, alpha=1.0, beta=50)     # 只管加亮
dark   = cv2.convertScaleAbs(gray, alpha=1.0, beta=-50)    # beta 负数 → 变暗
strong = cv2.convertScaleAbs(gray, alpha=1.5)              # 拉对比
weak   = cv2.convertScaleAbs(gray, alpha=0.5)              # 压对比

print("beta=+50  平均", round(bright.mean(), 2), "最小", bright.min(), "最大", bright.max())
print("beta=-50  平均", round(dark.mean(), 2), "最小", dark.min(), "最大", dark.max())
print("alpha=1.5 平均", round(strong.mean(), 2), "最小", strong.min(), "最大", strong.max())
print("alpha=0.5 平均", round(weak.mean(), 2), "最小", weak.min(), "最大", weak.max())

# 小数陷阱实证
small = np.array([200, 240, 250, 255, 10], dtype=np.uint8)
print("原始      ", small)
print("直接加 50 ", small + np.uint8(50))       # 危险写法：会超出 0~255 范围然后自动顺位回到错误的 0~255
print("convertScaleAbs(beta=50) ", cv2.convertScaleAbs(small, alpha=1.0, beta=50))

# ========== 段 4：equalizeHist 自动摊平 ==========
eq = cv2.equalizeHist(gray)
print("均衡前 平均", round(gray.mean(), 2), "标准差", round(gray.std(), 2),
      "最暗", gray.min(), "最亮", gray.max())
print("均衡后 平均", round(eq.mean(), 2), "标准差", round(eq.std(), 2),
      "最暗", eq.min(), "最亮", eq.max())

hist_eq = cv2.calcHist([eq], [0], None, [256], [0, 256])
print("均衡后直方图 最高的一根柱子 =", int(hist_eq.max()), " 在第", hist_eq.argmax(), "号桶")

# ========== 段 5：并排对比 + 回到真样本 ==========
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
axes[0].plot(hist)
axes[0].set_title("均衡前")
axes[0].set_xlabel("亮度")
axes[0].set_ylabel("像素个数")
axes[1].plot(hist_eq)
axes[1].set_title("均衡后")
axes[1].set_xlabel("亮度")
plt.tight_layout()             # 自动调整布局，避免文字重叠/被裁剪
plt.savefig("day30_均衡前后对比.png")
plt.close()
print("图已保存：day30_均衡前后对比.png")

# —— 回到你的编码器样本 ok1 ——
gray_ok = cv2.cvtColor(cv2.imread(OK1), cv2.COLOR_BGR2GRAY)
hist_ok = cv2.calcHist([gray_ok], [0], None, [256], [0, 256])
print("ok1 灰度 shape =", gray_ok.shape, " 像素总数 =", gray_ok.size)
print("ok1 平均亮度 =", round(gray_ok.mean(), 2),
      " 最暗 =", gray_ok.min(), " 最亮 =", gray_ok.max())
print("ok1 直方图峰值 =", int(hist_ok.max()), " 在第", hist_ok.argmax(), "号桶")
print("ok1 暗端(0-31)像素占比 =",
      round(float(hist_ok[:32].sum()) / gray_ok.size * 100, 2), "%")