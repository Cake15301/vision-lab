import cv2
import numpy as np

IMG = "D:/AI/zcode files/vision-lab/practice/day26_assets/ok1.jpg"        # 你的编码器样本
GH = "D:/AI/zcode files/vision-lab/practice/day26_assets/grace_hopper.jpg"  # 人像小图

# ========== 段 0：灰度化是加权，不是平均 ==========
img = cv2.imread(IMG)
gh = cv2.imread(GH)
print("ok1 彩色 shape =", img.shape, "  内存 =", img.nbytes, "字节 ≈", round(img.nbytes / 1024 / 1024, 1), "MB")
gray_ok = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("ok1 灰度 shape =", gray_ok.shape, "  内存 =", gray_ok.nbytes, "字节 ≈", round(gray_ok.nbytes / 1024 / 1024, 1), "MB")
print("彩色是灰度的几倍 =", round(img.nbytes / gray_ok.nbytes, 2))

b, g, r = gh[300, 256]
print("人像取一个像素 gh[300,256] 的 BGR =", b, g, r)
print("如果简单平均 (B+G+R)/3 =", round((int(b) + int(g) + int(r)) / 3, 4))
print("cv2 加权公式 0.114B+0.587G+0.299R =", round(0.114 * int(b) + 0.587 * int(g) + 0.299 * int(r), 4))
gray_gh = cv2.cvtColor(gh, cv2.COLOR_BGR2GRAY)
print("cv2 亲口给出的灰度值 =", gray_gh[300, 256])

print("ok1 中心像素 =", img[1824, 2736], " → 灰度化后 =", gray_ok[1824, 2736])
print("ok1 灰度平均亮度 =", round(gray_ok.mean(), 4), "  人像灰度平均亮度 =", round(gray_gh.mean(), 4))

# ========== 段 1：threshold 的机制与两个返回值 ==========
patch = gh[240:340, 200:300]                 # 从人像上切 100 行 × 100 列
gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
print("小块 shape =", gray_patch.shape, "  类型 =", gray_patch.dtype)

ret, bw = cv2.threshold(gray_patch, 127, 255, cv2.THRESH_BINARY)
print("拿回两样：ret =", ret, "   bw.shape =", bw.shape, "   bw.dtype =", bw.dtype)
print("bw 里出现的值 =", np.unique(bw))
print("白(255) 个数 =", int((bw == 255).sum()), "  黑(0) 个数 =", int((bw == 0).sum()), "  合计 =", bw.size)
print("小块原图里最小值 =", int(gray_patch.min()), " 最大值 =", int(gray_patch.max()), " 平均 =", round(gray_patch.mean(), 4))
print("原图里 >127 的像素个数 =", int((gray_patch > 127).sum()))
print("白像素个数 == 原图 >127 的个数？", int((bw == 255).sum()) == int((gray_patch > 127).sum()))

tiny = np.array([[126, 127, 128]], dtype=np.uint8)
print("三连号 126/127/128 过 127 门槛 →", cv2.threshold(tiny, 127, 255, cv2.THRESH_BINARY)[1].ravel())  #cv2.threshold(...) 返回的是(阈值, 二值化图像)
# 取出来的 dst 还是二维的 [[0, 0, 255]]。.ravel() 是 NumPy 的方法，作用是把数组展平成一维：[[0, 0, 255]].ravel()  →  [0, 0, 255]

# ========== 段 2：五种阈值类型 ==========
ramp = np.tile(np.arange(0, 256, dtype=np.uint8), (40, 1))   # 40 行，每行 0..255
print("亮度尺 shape =", ramp.shape, "  第 0 行前 8 个数 =", ramp[0, :8])
probe_cols = [120, 126, 127, 128, 200]
print("原值        ", "  ".join(f"{c:>4d}" for c in probe_cols))    # c:>4d：打印c，宽度4，十进制整数，右对齐
for name, t in [("THRESH_BINARY", cv2.THRESH_BINARY), ("THRESH_BINARY_INV", cv2.THRESH_BINARY_INV),
                ("THRESH_TRUNC", cv2.THRESH_TRUNC), ("THRESH_TOZERO", cv2.THRESH_TOZERO),
                ("THRESH_TOZERO_INV", cv2.THRESH_TOZERO_INV)]:
    _, out = cv2.threshold(ramp, 127, 255, t)
    print(f"{name:<20s}", "  ".join(f"{int(v):>4d}" for v in out[0, probe_cols]))   # " ".join(...)：用两个空格把它们黏成一个字符串

# ========== 段 3：回到你自己的编码器代码 thr=200 ==========
for thr in (127, 200):
    ret2, bw_ok = cv2.threshold(gray_ok, thr, 255, cv2.THRESH_BINARY)
    n_white = int((bw_ok == 255).sum())
    print(f"thr={thr}: ret={ret2}  白像素={n_white} ({round(n_white / bw_ok.size * 100, 6)}%)  "
          f"黑像素={int((bw_ok == 0).sum())}")
print("全图像素数 =", gray_ok.size, "= 3648 × 5472")    

# ========== 段 4：Otsu 自动阈值 ==========
ret_o, bw_ok_o = cv2.threshold(gray_ok, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print("ok1 的 Otsu 挑出 =", ret_o, "  白像素占比 =", round((bw_ok_o == 255).mean() * 100, 4), "%")
ret_og, bw_gh_o = cv2.threshold(gray_gh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print("人像的 Otsu 挑出 =", ret_og, "  白像素占比 =", round((bw_gh_o == 255).mean() * 100, 4), "%")

# ===== 补课OTSU原理：亲手扫 256 条线，看 127 是不是最高分 =====
hist = cv2.calcHist([gray_ok], [0], None, [256], [0, 256]).ravel()   # 直方图：每个亮度的像素个数
total = gray_ok.size
sum_all = float(np.dot(np.arange(256), hist))                        # 全图亮度总和
scores = []
w_dark, sum_dark = 0.0, 0.0
for t in range(256):
    w_dark += hist[t]                     # 亮度 <= t 的像素个数（暗堆）
    w_bright = total - w_dark             # 其余（亮堆）
    if w_dark == 0 or w_bright == 0:
        continue
    sum_dark += t * hist[t]
    mu_dark = sum_dark / w_dark                       # 暗堆平均亮度
    mu_bright = (sum_all - sum_dark) / w_bright       # 亮堆平均亮度
    score = (w_dark / total) * (w_bright / total) * (mu_dark - mu_bright) ** 2
    scores.append((t, score))
best_t, best_s = max(scores, key=lambda x: x[1])
print("手写扫出来的最高分那条线 =", best_t, "  得分 =", round(best_s, 1))
print("跟 cv2 的 Otsu 一致吗？", best_t == int(ret_o))
print("得分最高的 5 条线（阈值, 得分）：", [(t, round(float(s), 1)) for t, s in sorted(scores, key=lambda x: -x[1])[:5]])
for probe in (60, 100, 127, 160, 200):
    print(f"  阈值 {probe:>3d} 这条线的得分 =", round(float(dict(scores)[probe]), 1))

# ========== 段 5：自适应阈值治光照不均 ==========
h, w = 300, 600
floor = np.tile(np.linspace(40, 210, w), (h, 1))            # 越往右越亮的地板，linspace(start, stop, num) 生成等差数列，tile(h, 1) 是把数组复制h行,复制1列(列不变)
uneven = floor.copy()
uneven[100:130, 100:130] = floor[100:130, 100:130] - 60     # 暗区里一块深色小物体
uneven[100:130, 430:460] = floor[100:130, 430:460] - 60     # 亮区里同样的深色小物体
uneven = np.clip(uneven, 0, 255).astype(np.uint8)           # np.clip 把越界值夹到 [0, 255]（减法可能出现负值）, astype(np.uint8) 是把数组转换为 uint8 类型，即8位无符号整数
print("地板亮度：左端 =", int(uneven[0, 0]), " 右端 =", int(uneven[0, -1]), "  物体 30×30、比脚下地板深 60")
print("暗区物体亮度 =", int(uneven[115, 115]), "  亮区物体亮度 =", int(uneven[115, 445]))

_, g_bw = cv2.threshold(uneven, 127, 255, cv2.THRESH_BINARY)
a_good = cv2.adaptiveThreshold(uneven, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 101, 10) # 101：blockSize（邻域窗口尺寸），必须是奇数，即每个像素看它周围 101×101 的区域
a_bad = cv2.adaptiveThreshold(uneven, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 10)   # 10：C 常数，实际阈值 = 邻域均值 − 10

regions = [("暗区地板", (0, 60, 0, 60)), ("暗区物体", (105, 125, 105, 125)),
           ("亮区地板", (0, 60, 540, 600)), ("亮区物体", (105, 125, 435, 455))]
print(f"{'':<22s}" + "".join(f"{n:>12s}" for n, _ in regions))
for name, m in [("全局 thr=127", g_bw), ("自适应(bs=101,C=10)", a_good), ("自适应(bs=11,C=10)", a_bad)]:
    cells = [f"{(m[r0:r1, c0:c1] == 255).mean() * 100:>11.2f}%" for _, (r0, r1, c0, c1) in regions]
    print(f"{name:<22s}" + "".join(cells))

# ===== 亮物体版：模拟你编码器那种「亮缺陷」=====
bright = floor.copy()
bright[100:130, 100:130] = floor[100:130, 100:130] + 60
bright[100:130, 430:460] = floor[100:130, 430:460] + 60
bright = np.clip(bright, 0, 255).astype(np.uint8)

_, b_global = cv2.threshold(bright, 200, 255, cv2.THRESH_BINARY)
b_neg = cv2.adaptiveThreshold(bright, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 101, -15)
b_pos = cv2.adaptiveThreshold(bright, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 101, 10)
for name, m in [("全局 thr=200", b_global), ("自适应(bs=101,C=-15)", b_neg), ("自适应(bs=101,C=+10)", b_pos)]:
    cells = [f"{(m[r0:r1, c0:c1] == 255).mean() * 100:>11.2f}%" for _, (r0, r1, c0, c1) in regions]
    print(f"{name:<22s}" + "".join(cells))

# ===== 拿真图收尾：你自己那张 ok1 =====
for C in (10, -5, -15, -30):
    m = cv2.adaptiveThreshold(gray_ok, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, C)
    print(f"自适应(bs=51,C={C:>3d}): 白占比 = {round((m == 255).mean() * 100, 4)}%")

# ===== 拼 2×2 对比板 =====
board_top = np.hstack([uneven, g_bw])
board_bot = np.hstack([a_good, a_bad])
board = np.vstack([board_top, board_bot])
print("board.shape =", board.shape)
cv2.imshow("day29_threshold", board)
print("窗口出来了吗？（看完按键盘任意键）")
cv2.waitKey(0)
cv2.destroyAllWindows()
print("窗口已关闭")