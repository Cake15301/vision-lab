from pathlib import Path

p = Path("D:/AI/zcode files/vision-lab")
print(type(p).__name__)          # 我到底是个什么品种
for c in type(p).__mro__:        # __mro__ = 家谱（从下往上，一直列到 object）
    print("   ", c.__name__)

print(len(dir(p))) 

from torchvision.datasets import ImageFolder as TvImageFolder   # 先给正牌货起个别名，免得跟咱自己的撞名
print([c.__name__ for c in TvImageFolder.__mro__])

class FolderBase:
    def __init__(self, folder, pattern):
        self.folder = Path(folder)
        self.pattern = pattern
        self.files = list(self.folder.glob(pattern))

    def __len__(self):
        return len(self.files)

    def __str__(self):
        return f"{type(self).__name__}：{len(self.files)} 个文件"


base = FolderBase("D:/AI/zcode files/YOLO入门学习/datasets/coco8/labels", "**/*.txt")
print(base)
print(len(base))

class LabelStats(FolderBase):
    def __init__(self, labels_dir):
        super().__init__(labels_dir, "**/*.txt")     # ← 先请师父打底

    def count_by_class(self):
        counts = {}
        for path in self.files:                      # self.files 是师父扫好存下的
            text = path.read_text(encoding="utf-8")
            for line in text.splitlines():
                if not line.strip():
                    continue
                cls = int(line.split()[0])
                counts[cls] = counts.get(cls, 0) + 1
        return counts

    def __str__(self):
        counts = self.count_by_class()
        return f"LabelStats：{len(self)} 个文件，{len(counts)} 个类别，{sum(counts.values())} 个目标"


s = LabelStats("D:/AI/zcode files/YOLO入门学习/datasets/coco8/labels")
print(s)
print(len(s))
print(s.count_by_class())

class ImageFolder(FolderBase):
    def __init__(self, images_dir):
        super().__init__(images_dir, "**/*.jpg")

    def list_images(self):
        return [str(f) for f in self.files]

imgs = ImageFolder("D:/AI/zcode files/YOLO入门学习/datasets/coco8/images")
print(imgs)
print(len(imgs))
print(imgs.list_images())

print(isinstance(imgs, FolderBase))     # 徒弟算不算师父家的人
print(isinstance(base, ImageFolder))    # 反着问：师父算不算徒弟家的人