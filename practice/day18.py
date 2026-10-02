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

good = 0
bad = 0
counts = {}
bad_file = []
for p in txt_files:
    try:
        text = p.read_text(encoding="utf-8")
        lines = [line for line in text.splitlines() if line.strip()]
        for line in lines:
            cls = int(line.split()[0])
            if cls in counts:
                counts[cls] += 1
            else:
                counts[cls] = 1
        good += 1
    except ValueError as e:
        print(f"跳过坏文件 {p.name}：{e}")
        bad += 1
        bad_file.append(p.name)
print(f"扫描完毕：好的 {good} 个，坏的 {bad} 个，一共 {good + bad} 个文件")
print(f"类别数 {len(counts)}，目标总数 {sum(counts.values())}")

with open("类别分布.csv", "w", encoding="utf-8") as f:
    f.write("class_id,class_name,count\n")
    for cls in sorted(counts):
        f.write(f"{cls},{names[cls]},{counts[cls]}\n")
print("CSV 已写出")

with open("坏文件.csv", "w", encoding="utf-8") as f:
    for name in bad_file:
        f.write(name + "\n")
print("坏文件列表已写出")