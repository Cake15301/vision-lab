from pathlib import Path

folder = Path("D:/AI/zcode files/YOLO入门学习/datasets/coco8")

print(folder)
print(folder.exists())

labels = folder / "labels"
images = folder / "images"

print(labels)
print(images)

txt_files = list(labels.glob("**/*.txt"))
print(len(txt_files))

total = 0
for p in txt_files:
    print(p.name)
    text = p.read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if line.strip()]
    total += len(lines)

print(f"标注文件 {len(txt_files)} 个")
print(f"目标总数 {total} 个")

image_files = list(images.glob("**/*.jpg"))
print(f"图片 {len(image_files)} 张")
print(f"平均每张 {total / len(image_files)} 个目标")

