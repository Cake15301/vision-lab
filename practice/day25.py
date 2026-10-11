import numpy as np
from PIL import Image
from pathlib import Path
import cv2

# ========== 段 0 ==========
a = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]])
print("a.sum() =", a.sum())
print("a.mean() =", a.mean())
print("a.max() =", a.max())
print("a.min() =", a.min())

img = np.array(Image.open("D:/AI/zcode files/YOLO入门学习/datasets/coco8/images/val/000000000036.jpg"))
print("这张图的 mean =", round(img.mean(), 2))      #round(..., 2) 四舍五入，保留2位小数
print("这张图的 max =", img.max(), " min =", img.min())

# ========== 段 1 ==========
print("a.sum(axis=0) =", a.sum(axis=0))
print("a.sum(axis=1) =", a.sum(axis=1))
print("a.mean(axis=0) =", a.mean(axis=0))
print("图上三个颜色通道各自平均 =", img.mean(axis=(0, 1)))

# ========== 段 2 ==========
scores = np.array([0.2, 0.05, 0.6, 0.1, 0.05])
print("scores =", scores)
print("scores.argmax() =", scores.argmax())     # 拿回来的是位置！

# ========== 段 3 ==========
row = np.array([1, 0, -1, 0])
print("a + row =")
print(a + row)

print("img * 0.5 的 dtype:", (img * 0.5).dtype)
img_half = (img * 0.5).astype(np.uint8)
print("转回 uint8 后 dtype:", img_half.dtype)
print("img[0, 0] =", img[0, 0], " 半暗后 =", img_half[0, 0])

# ========== 段 4 ==========
P_EN = "D:/python/Lib/site-packages/matplotlib/mpl-data/sample_data/grace_hopper.jpg"
img_pil = np.array(Image.open(P_EN))
img_cv = cv2.imread(P_EN)
print("PIL  读的 [0,0] =", img_pil[0, 0])
print("cv2  读的 [0,0] =", img_cv[0, 0])
img_fixed = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
print("转回 RGB 后 [0,0] =", img_fixed[0, 0])
print("转完跟 PIL 全图一致:", (img_fixed == img_pil).all())

img_cn = cv2.imread("D:/AI/zcode files/YOLO入门学习/datasets/coco8/images/val/000000000036.jpg")
print("cv2 读中文路径得到:", img_cn)


# ========== 段 6：本周大作业 ==========
class ImageFolder:
    """一个最小的图片文件夹加载器——本周六天零件的合体。"""

    def __init__(self, images_dir: str) -> None:
        self.images_dir = Path(images_dir)
        self.files: list[Path] = list(self.images_dir.glob("**/*.jpg"))

    def __len__(self) -> int:
        return len(self.files)

    def __str__(self) -> str:
        return f"ImageFolder: {len(self)} 张图，目录 {self.images_dir}"

    @property       #把一个"方法"伪装成"属性"，让调用时不用写括号。
    def names(self) -> list[str]:
        return [p.name for p in self.files]

    def load(self, idx: int) -> np.ndarray:
        return np.array(Image.open(self.files[idx]))

folder = ImageFolder("D:/AI/zcode files/YOLO入门学习/datasets/coco8/images/val")
print(len(folder))
print(folder.names)
print(folder)
img0 = folder.load(0)
print("load(0):", type(img0).__name__, img0.shape, img0.dtype)
print("load(1) shape =", folder.load(1).shape)