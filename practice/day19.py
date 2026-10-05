from pathlib import Path

p = Path("D:/AI/zcode files/vision-lab")
print(type(p))
print(type("hello"))
print(type(3))

methods = [m for m in dir(p) if not m.startswith("_")]
print("Path 的公开方法数:", len(methods))
print("前 10 个:", methods[:10])

class Dog:
    def __init__(self,name,age):
        self.name = name
        self.age = age

    def greet(self):
        return f"汪！我叫{self.name}，今年{self.age}岁"

    def birthday(self):
        self.age += 1
        return f"汪！我过生日了，今年{self.age}岁"

    def human_age(self):
        return self.age * 7
    

d = Dog("旺财", 2)
print(d.greet())
d.birthday()
print(d.greet())
d2 = Dog("小黑", 5)
print(d2.greet())
print("旺财:", d.name, d.age, "| 小黑:", d2.name, d2.age)
print(d.human_age())

class LabelStats:
    def __init__(self, labels_dir):
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

    def top_class(self):
        counts = self.count_by_class()
        best = max(counts, key=counts.get)
        return (best, counts[best])

s = LabelStats("D:/AI/zcode files/YOLO入门学习/datasets/coco8/labels")
print("标注文件数:", len(s.txt_files))
counts = s.count_by_class()
print("类别数:", len(counts))
print("分布:", counts)
print("目标总数:", sum(counts.values()))
best = max(counts, key=counts.get)
print("最多类别:", best, "共", counts[best], "个")
rare = [c for c in counts if counts[c] == 1]
print("只出现1次的类别:", sorted(rare))
print(s.top_class())