from pathlib import Path

folder = Path("D:/AI/zcode files/YOLO入门学习/datasets/coco8")
labels = folder / "labels"
txt_files = list(labels.glob("**/*.txt"))

names = {
    0: "person",
    16: "dog",
    17: "horse",
    20: "elephant",
    22: "zebra",
    23: "giraffe",
    25: "umbrella",
    45: "bowl",
    49: "orange",
    50: "broccoli",
    58: "potted plant",
    75: "vase",
}

counts = {}
for p in txt_files:
    text = p.read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if line.strip()]
    for line in lines:
        cls = int(line.split()[0])
        if cls in counts:
            counts[cls] += 1
        else:
            counts[cls] = 1

with open("类别分布.csv", "w", encoding="utf-8") as f:
    f.write("class_id,class_name,count\n")
    for cls in sorted(counts):
        f.write(f"{cls},{names[cls]},{counts[cls]}\n")
print("CSV 已写出")

cls_list = sorted(counts)
names_list = [names[c] for c in cls_list]
counts_list = [counts[c] for c in cls_list]
print(names_list)
print(counts_list)

import matplotlib.pyplot as plt

plt.bar(names_list, counts_list)
plt.savefig("柱状图_第一版.png")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

plt.title("coco8 类别分布")
plt.xlabel("类别")
plt.ylabel("个数")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("柱状图_第二版.png")
print("第二版已保存")