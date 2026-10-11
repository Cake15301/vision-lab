import numpy as np
from PIL import Image
import time

# ========== 段 0 ==========
py_list = [1, 2, 3]
print("列表 + 列表:", py_list + [4, 5, 6])

arr = np.array([1, 2, 3])
print("数组 + 数组:", arr + np.array([4, 5, 6]))
print("数组 * 2   :", arr * 2)

# ========== 段 1 ==========
a = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]])
print("a =")
print(a)
print("a.shape =", a.shape)
print("a.dtype =", a.dtype)
print("a.ndim  =", a.ndim)

z = np.zeros((2, 3))
print("zeros shape:", z.shape, "dtype:", z.dtype)

r = np.arange(5, 11)
print("arange(5, 11) =", r)

# ========== 段 2 ==========
print("a[1, 2] =", a[1, 2])
print("a[2, 0] =", a[2, 0])
print("a[0]    =", a[0])
print("a[1][2] =", a[1][2])

# ========== 段 3 ==========
print("a[0:2, 1:3] =")
print(a[0:2, 1:3])
print("a[:, 3] =", a[:, 3])
print("a[1, :] =", a[1, :])

# ========== 段 4 ==========
img = np.array(Image.open("D:/AI/zcode files/YOLO入门学习/datasets/coco8/images/val/000000000036.jpg"))
print("img.shape =", img.shape)
print("img.dtype =", img.dtype)
print("img[0, 0] =", img[0, 0])
print("img[0, 0, 0] =", img[0, 0, 0])

# ========== 段 5 ==========
block = img[100:140, 200:260]
print("block.shape =", block.shape)

img2 = img.copy()
img2[:60, :60] = 0
print("改后左上角像素:", img2[0, 0])
print("改后值范围:", img2.min(), "~", img2.max())

# ========== 段 6 ==========
print("a > 5 =")
print(a > 5)
print("(a > 5).sum() =", (a > 5).sum())

big = np.arange(1_000_000)
t0 = time.time()
s = 0
for x in big:
    s += x
t1 = time.time()
s2 = big.sum()
t2 = time.time()
print("for 循环求和:", s, "耗时", round(t1 - t0, 4), "秒")
print("numpy 求和  :", s2, "耗时", round(t2 - t1, 4), "秒")

# ========== 段 7 ==========
c = a[0:2, 0:2]
print("刚切出来的 c =")
print(c)
c[0, 0] = 999
print("改了 c[0,0] 之后，a[0,0] =", a[0, 0])

d = a.copy()
d[0, 0] = 555
print("改了复印件 d[0,0]，a[0,0] =", a[0, 0])