import cv2
import numpy as np

# ========== 段 0：ok1 第一次旋转 ==========
img = cv2.imread("D:/AI/zcode files/vision-lab/practice/day26_assets/ok1.jpg")
print("原图 shape =", img.shape)
M_ok = cv2.getRotationMatrix2D(center=(2736, 1824), angle=45, scale=1.0)    # center 报的也是 (x, y)，就是（列，行）
rot_ok = cv2.warpAffine(img, M_ok, (5472, 3648))
print("rot_ok.shape =", rot_ok.shape)
print("左上角 rot_ok[0, 0] =", rot_ok[0, 0])
print("右上角 rot_ok[0, -1] =", rot_ok[0, -1])
print("中心 rot_ok[1824, 2736] =", rot_ok[1824, 2736], " 原图同位置 =", img[1824, 2736])

# ========== 段 1：操作单 M 长什么样 ==========
gh = cv2.imread("D:/AI/zcode files/vision-lab/practice/day26_assets/grace_hopper.jpg")
print("gh.shape =", gh.shape)
M = cv2.getRotationMatrix2D(center=(256, 300), angle=45, scale=1.0)
print(M)

# 操作单 M 矩阵的原理
alpha = np.cos(np.radians(45))
beta = np.sin(np.radians(45))
print("cos45 =", round(alpha, 8), " sin45 =", round(beta, 8))
tx = (1 - alpha) * 256 - beta * 300
ty = beta * 256 + (1 - alpha) * 300
print("自己算的 tx =", round(tx, 8), " ty =", round(ty, 8))

# ========== 段 2：+90 度往哪边转 ==========
gh2 = cv2.imread("D:/AI/zcode files/vision-lab/practice/day26_assets/grace_hopper.jpg")
gh2[50:80, 50:80] = (0, 0, 255)
print("红块原来在：行 50 到 79，列 50 到 79")
M90 = cv2.getRotationMatrix2D(center=(256, 300), angle=90, scale=1.0)
rot90 = cv2.warpAffine(gh2, M90, (512, 600))
ys, xs = np.where((rot90[:, :, 2] == 255) & (rot90[:, :, 1] == 0) & (rot90[:, :, 0] == 0))  # 从0开始计数,所以2是红通道，1是绿通道，0是蓝通道
print("红块新位置：行", ys.min(), "到", ys.max(), "  列", xs.min(), "到", xs.max())

# ========== 段 3：黑角填 114 灰 ==========
rot45 = cv2.warpAffine(gh, M, (512, 600))
rot45_gray = cv2.warpAffine(gh, M, (512, 600), borderValue=(114, 114, 114))
print("不填色：左上角 =", rot45[0, 0], "  右下角 =", rot45[-1, -1])
print("填 114：左上角 =", rot45_gray[0, 0], "  右下角 =", rot45_gray[-1, -1])

# ========== 段 4：三点定形（斜切） ==========
src = np.float32([[0, 0], [511, 0], [0, 599]])
dst = np.float32([[50, 0], [561, 0], [0, 599]])
M2 = cv2.getAffineTransform(src, dst)
print("src =", src.tolist())
print("dst =", dst.tolist())
print(M2)
lean = cv2.warpAffine(gh, M2, (512, 600))
print("lean.shape =", lean.shape)

# ========== 段 5：拼 2x2 对比板 ==========
board_top = np.hstack([gh, rot45])          # 横拼
board_bot = np.hstack([rot45_gray, lean])   
board = np.vstack([board_top, board_bot])   # 竖拼
print("board.shape =", board.shape)
cv2.imshow("day28_rotation", board)
print("窗口出来了吗？（看完按键盘任意键）")
cv2.waitKey(0)
cv2.destroyAllWindows()
print("窗口已关闭")