import cv2
import numpy as np

# ========== 段 0：resize 第一试 ==========
img = cv2.imread("D:/AI/zcode files/vision-lab/practice/day26_assets/ok1.jpg")
print("原图 shape =", img.shape)
small = cv2.resize(img, (640, 480))
print("resize(img, (640, 480)) 后 shape =", small.shape)    # 报给resize()的尺寸是(宽, 高),.shape是(高, 宽, 通道)!

# ========== 段 1：比例缩放与内存账 ==========
half = cv2.resize(img, None, fx=0.5, fy=0.5)    #None表示不指定尺寸，而是用 fx, fy 来指定缩放比例
print("fx=fy=0.5 后 shape =", half.shape)
print("half.nbytes =", half.nbytes, " 约合", round(half.nbytes / 1024 / 1024, 1), "MB")
thumb = cv2.resize(img, (640, 480))
print("640x480 nbytes =", thumb.nbytes)

# ========== 段 2：插值对比 ==========
gh = cv2.imread("D:/AI/zcode files/vision-lab/practice/day26_assets/grace_hopper.jpg")
print("gh.shape =", gh.shape)
patch = gh[250:254, 250:254]
print("4x4 小块的蓝通道 =")
print(patch[:, :, 0])       #取所有行，所有列，第零层(蓝通道)
big_near = cv2.resize(patch, (400, 400), interpolation=cv2.INTER_NEAREST)
big_lin = cv2.resize(patch, (400, 400), interpolation=cv2.INTER_LINEAR)
print("最近邻放大后 [100,100] =", big_near[100, 100])
print("线性放大后   [100,100] =", big_lin[100, 100])
#最近邻：取最近的像素值直接抄过来，不考虑插值；线性：取最近的像素值的插值加权平均，更平滑

# ========== 段 3：切片裁剪与视图坑 ==========
img3 = cv2.imread("D:/AI/zcode files/vision-lab/practice/day26_assets/ok1.jpg")
center = img3[1224:2424, 2136:3336]
print("center.shape =", center.shape)
print("改之前 img3[1224, 2136] =", img3[1224, 2136])
center[0:600, 0:600] = 255
print("把 center 刷白之后 img3[1224, 2136] =", img3[1224, 2136])
img4 = cv2.imread("D:/AI/zcode files/vision-lab/practice/day26_assets/ok1.jpg")
center2 = img4[1224:2424, 2136:3336].copy()
center2[0:600, 0:600] = 255
print("copy 版刷白之后 img4[1224, 2136] =", img4[1224, 2136])
#   切片后只刷第一行/列/全部，代码为:[1]=255/ [:,1]=255/ [:,:]=255

# ========== 段 4：翻转 ==========
gh_v = cv2.flip(gh, 0)
gh_h = cv2.flip(gh, 1)
gh_both = cv2.flip(gh, -1)
print("gh[0, 0] =", gh[0, 0], "  gh_h[0, -1] =", gh_h[0, -1])   # [0，0] 是左上角，[0, -1] 是右上角，-1表示最后一个元素
print("flip(1) 等于切片 [:, ::-1] 吗？", np.array_equal(gh_h, gh[:, ::-1])) # ::-1 表示所有行或列倒着走
print("flip(0) 等于切片 [::-1, :] 吗？", np.array_equal(gh_v, gh[::-1, :]))
print("flip(-1) 等于切片 [::-1, ::-1] 吗？", np.array_equal(gh_both, gh[::-1, ::-1]))

# ========== 段 5：拼对比图 ==========
row = np.hstack([gh, gh_h, gh_both])
print("横拼 shape =", row.shape)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("翻转前后 mean：", round(gray.mean(), 1), "vs", round(cv2.flip(gray, 1).mean(), 1))
cv2.imshow("day27_flips", row)
print("窗口出来了吗？（看完按键盘任意键）")
cv2.waitKey(0)
cv2.destroyAllWindows()
print("窗口已关闭")