from pathlib import Path

for nm in ["name", "suffix", "stem", "parent", "exists", "glob"]:
    obj = getattr(Path, nm)          # 从图纸上把这个名字取出来，看看它是什么品种
    print(nm, "->", type(obj).__name__)

f = Path("D:/AI/zcode files/vision-lab/practice/day22.py")
print(f.name, "|", f.stem, "|", f.suffix, "|", f.parent)

props, funcs = [], []
for nm in dir(Path):
    if nm.startswith("_"):           # 下划线开头的先不管（Day19 见过）
        continue
    if isinstance(getattr(Path, nm), property):     # 是"属性"这个品种吗
        props.append(nm)
    elif callable(getattr(Path, nm)):               # 是"能喊的"吗
        funcs.append(nm)

print("Path 上的公开属性（不用括号）:", len(props))
print(props)
print("Path 上的公开方法（要括号）:", len(funcs))
print(funcs[:12])

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

    @property                                   # ← 今天的新零件
    def total(self):
        return sum(self.count_by_class().values())


s = LabelStats("D:/AI/zcode files/YOLO入门学习/datasets/coco8/labels")
print("方法（带括号）:", sum(s.count_by_class().values()))
print("属性（不带括号）:", s.total)
print("s 身上有 total 这个名字吗:", hasattr(s, "total"))

class ImageFolder:
    def __init__(self, images_dir):
        self.images_dir = Path(images_dir)
        self.files = list(self.images_dir.glob("**/*.jpg"))

    def __len__(self):
        return len(self.files)

    def __str__(self):
        return f"ImageFolder：{len(self)} 张图"

    @property
    def names(self):
        return [p.name for p in self.files]     # ← p.name 就是第 0 段那个属性

imgs = ImageFolder("D:/AI/zcode files/YOLO入门学习/datasets/coco8/images")
print(imgs)
print(len(imgs))
print("老写法:", [p.name for p in imgs.files])
print("新写法:", imgs.names)

from dataclasses import dataclass      # 补在文件最上面 import 区

# ---- 老实版 ----
class BBoxOld:
    def __init__(self, cls, cx, cy, w, h):
        self.cls = cls
        self.cx = cx
        self.cy = cy
        self.w = w
        self.h = h

# ---- dataclass 版 ----
@dataclass
class BBox:
    cls: int
    cx: float
    cy: float
    w: float
    h: float

    @property
    def area(self):
        return self.w * self.h

old = BBoxOld(45, 0.479492, 0.688771, 0.955609, 0.5955)
new = BBox(45, 0.479492, 0.688771, 0.955609, 0.5955)
print("老实版 print():", old)
print("dataclass print():", new)
print("老实版字段:", old.cls, old.cx, old.cy, old.w, old.h)
print("dataclass 字段:", new.cls, new.cx, new.cy, new.w, new.h)
print("老实版比相等:", BBoxOld(45, 0.479492, 0.688771, 0.955609, 0.5955) == BBoxOld(45, 0.479492, 0.688771, 0.955609, 0.5955))
print("dataclass 比相等:", BBox(45, 0.479492, 0.688771, 0.955609, 0.5955) == BBox(45, 0.479492, 0.688771, 0.955609, 0.5955))
print(new.area)  

@dataclass
class BBox2:
    cls: int
    cx: float
    cy: float
    w: float
    h: float
    conf: float = 1.0          # ← 有默认值，必须排在没默认值的后面

b1 = BBox2(45, 0.479492, 0.688771, 0.955609, 0.5955)          # 不给 conf
b2 = BBox2(0, 0.5, 0.5, 0.3, 0.3, 0.87)                       # 给了 conf
print("没给 conf 的:", b1)
print("给了 conf 的:", b2)
print("两者相等吗:", b1 == b2)

