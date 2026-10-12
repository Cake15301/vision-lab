import cv2
import numpy as np

# ========== 段 0：读第一张真图 ==========
img = cv2.imread("D:/AI/zcode files/vision-lab/practice/day26_assets/ok1.jpg")
print("读到了吗？", img is not None)
print("img.shape =", img.shape)
print("img.dtype =", img.dtype)

# ========== 段 1：解剖 ==========
print("img[0, 0] =", img[0, 0])
print("高 =", img.shape[0], " 宽 =", img.shape[1], " 通道 =", img.shape[2])

# ========== 段 2：位深与内存 ==========
print("img.nbytes =", img.nbytes)
print("约合", round(img.nbytes / 1024 / 1024, 1), "MB")
small = np.zeros((1080, 1920, 3), np.uint8)
print("1920x1080 彩色图 nbytes =", small.nbytes)

# ========== 段 3：颜色空间 ==========
gh = cv2.imread("D:/AI/zcode files/vision-lab/practice/day26_assets/grace_hopper.jpg")
print("gh[100, 100] BGR =", gh[100, 100])
gh_rgb = cv2.cvtColor(gh, cv2.COLOR_BGR2RGB)
gh_hsv = cv2.cvtColor(gh, cv2.COLOR_BGR2HSV)
print("gh[100, 100] RGB =", gh_rgb[100, 100])
print("gh[100, 100] HSV =", gh_hsv[100, 100])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("gray.shape =", gray.shape)
print("gray.mean() =", round(gray.mean(), 1))

# ========== 段 4：存出去读回来 ==========
saved = cv2.imwrite("D:/AI/zcode files/vision-lab/practice/day26_assets/ok1_gray.jpg", gray)
print("存成功了吗？", saved)
back = cv2.imread("D:/AI/zcode files/vision-lab/practice/day26_assets/ok1_gray.jpg")
print("默认读回来 shape =", back.shape)
back2 = cv2.imread("D:/AI/zcode files/vision-lab/practice/day26_assets/ok1_gray.jpg", cv2.IMREAD_GRAYSCALE)
print("IMREAD_GRAYSCALE 读回来 shape =", back2.shape)
print("back[1, 1] =", back[1, 1], "  back2[1, 1] =", back2[1, 1])

# ========== 段 5：中文路径坑与解法 ==========
cn = "D:/AI/zcode files/2026-10-02_编码器缺陷检测页面与实验报告/web/samples/ok1.jpg"
bad = cv2.imread(cn)
print("中文路径 imread 读到：", bad)
data = np.fromfile(cn, np.uint8)
print("fromfile 拿到", data.size, "个字节")
img2 = cv2.imdecode(data, cv2.IMREAD_COLOR)
print("imdecode 后 shape =", img2.shape)
print("和英文路径读的原图一模一样吗？", np.array_equal(img, img2))
ok, buf = cv2.imencode(".jpg", gray)        # cv2.imencode() 返回 bool, np.ndarray,所以要拿两个变量接收，np.ndarray是个n维数组
buf.tofile("D:/AI/zcode files/vision-lab/practice/day26_assets/中间结果_灰度.jpg")
print("中文名文件存好了")

# ========== 段 6：imshow 看一眼 ==========
block = img[1200:2000, 1800:3000]
print("切片块 shape =", block.shape)
cv2.imshow("ok1_center", block)
print("窗口出来了吗？（现在去按键盘上任意一个键）")
cv2.waitKey(0)
cv2.destroyAllWindows()
print("窗口已关闭")