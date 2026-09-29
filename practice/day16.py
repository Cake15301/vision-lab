from pathlib import Path

folder = Path("D:/AI/zcode files/YOLO入门学习/datasets/coco8")
labels = folder / "labels"
txt_files = list(labels.glob("**/*.txt"))

p = txt_files[0]
text = p.read_text(encoding="utf-8")
lines = [line for line in text.splitlines() if line.strip()]
for line in lines:
    print(line)

line = lines[0]                  
print(line.split())              
print(line.split()[0])           
print(int(line.split()[0]))      
print(type(line.split()[0]))     
print(type(int(line.split()[0])))

counts = {}
for line in lines:
    cls = int(line.split()[0])
    if cls in counts:
        counts[cls] += 1
    else:
        counts[cls] = 1
print(counts)

counts = {}
total = 0
for p in txt_files:
    text = p.read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if line.strip()]
    for line in lines:
        cls = int(line.split()[0])
        if cls in counts:
            counts[cls] += 1
        else:
            counts[cls] = 1
        total += 1
print(f"目标总数 {total} 个")
for cls in sorted(counts):
    print(f"类别 {cls} 共 {counts[cls]} 个")

max_cls = None
max_n = 0
for cls in counts:
    if counts[cls] > max_n:
        max_n = counts[cls]
        max_cls = cls

print(f"样本最多的类别：{max_cls}，{max_n} 个")