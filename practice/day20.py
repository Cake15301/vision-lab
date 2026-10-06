from pathlib import Path

p = Path("D:/AI/zcode files/vision-lab")
print(p)

print(len("hello"))            
print(len([1, 2, 3]))   

names = dir(p)                 # dir() 你 Day19 用过：列出 p 身上的所有名字
print("__str__" in names)      # "X in Y" = 问：Y 里有没有 X？回 True 或 False
print("__len__" in names)

try:                           
    len(p)
except TypeError as e:
    print("Path 没有 __len__，炸了：", e)

class LabelStats:
    made = 0  # 类属性，记录 LabelStats 被实例化了多少次

    def __init__(self, labels_dir):
        LabelStats.made = LabelStats.made + 1
        self.labels_dir = Path(labels_dir)
        self.txt_files = list(self.labels_dir.glob("**/*.txt"))

    def count_by_class(self):
        counts = {}
        for p in self.txt_files:
            text = p.read_text(encoding="utf-8")
            for line in text.splitlines():
                if not line.strip():
                    continue
                cls = int(line.split()[0])
                counts[cls] = counts.get(cls, 0) + 1
        return counts

    def __len__(self):
        return len(self.txt_files)

    def __str__(self):
        counts = self.count_by_class()
        return f"LabelStats：{len(self.txt_files)} 个文件，{len(counts)} 个类别，{sum(counts.values())} 个目标"

s = LabelStats("D:/AI/zcode files/YOLO入门学习/datasets/coco8/labels")
print(s)          # 第一个病：丑
print(len(s))     # 第二个病：炸

a = LabelStats("D:/AI/zcode files/YOLO入门学习/datasets/coco8/labels")
b = LabelStats("D:/AI/zcode files/YOLO入门学习/datasets/coco8/labels")
print(LabelStats.made, a.made, b.made)
a.made = 100     # 故意试一下：这是"给 a 单独贴新标签"，不是"改图纸"
print(LabelStats.made, a.made, b.made)