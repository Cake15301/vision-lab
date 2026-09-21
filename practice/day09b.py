import os

path = "D:/AI/zcode files/vision-lab/notes"
names = os.listdir(path)
print(names)

md_files = 0
max_bytes = 0
max_name = ""

for name in names:
    if name.endswith(".md"):
        md_files = md_files + 1
    full_path = os.path.join(path,name)
    x = os.path.getsize(full_path)
    if x > max_bytes:
        max_bytes = x
        max_name = name
print(f"{max_name}是最大的文件,大小为{max_bytes}字节")

print(f"总共有{md_files}个.md文件")



