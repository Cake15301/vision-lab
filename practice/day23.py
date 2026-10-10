from pathlib import Path
import dataclasses
from dataclasses import dataclass

# ========== 段 0 ==========
class LabelStats:
    def __init__(self, labels_dir):
        self.labels_dir = Path(labels_dir)

    def count_by_class(self):
        counts = {}
        for p in self.labels_dir.glob("**/*.txt"):
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    c = int(line.split()[0])
                    counts[c] = counts.get(c, 0) + 1
        return counts

print("我昨天那个函数，注解表是:", LabelStats.count_by_class.__annotations__)

# ========== 段 1 ==========
def add(a: int, b: int) -> int:
    return a + b

print(add(1, 2))
print(add("东", "华"))
print(add.__annotations__)

# ========== 段 2 ==========
def f(nums: list[int], m: dict[str, int], name: str | None = None) -> tuple[int, ...]:
    return (1, 2, 3)

print(f.__annotations__)
print(type(list[int]))

# ========== 段 3 ==========
class LabelStats2:
    def __init__(self, labels_dir: str) -> None:
        self.labels_dir = Path(labels_dir)
        self.txt_files: list[Path] = list(self.labels_dir.glob("**/*.txt"))

    def count_by_class(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for p in self.txt_files:
            text: str = p.read_text(encoding="utf-8")
            for line in text.splitlines():
                if not line.strip():
                    continue
                cls: int = int(line.split()[0])
                counts[cls] = counts.get(cls, 0) + 1
        return counts


s = LabelStats2("D:/AI/zcode files/YOLO入门学习/datasets/coco8/labels")
print("带注解版算出来的总数:", sum(s.count_by_class().values()))
print("它的注解表:", LabelStats2.count_by_class.__annotations__)

# ========== 段 4 ==========
from pydantic import BaseModel

class BBox(BaseModel):
    cls: int
    cx: float
    cy: float
    w: float
    h: float

print(BBox(cls=45, cx=0.479, cy=0.689, w=0.956, h=0.596))   # 第一行
print(BBox(cls="猫", cx=0.479, cy=0.689, w=0.956, h=0.596))  # 第二行被门卫拦了

# ========== 段 5 ==========
code = """from __future__ import annotations
def g(x: list[int]) -> int:
    return 1
"""
ns = {}
exec(code, ns)
print(repr(add.__annotations__["a"]))
print(repr(ns["g"].__annotations__["x"]))