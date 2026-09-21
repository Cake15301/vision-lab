import os

folder = "D:/AI/zcode files/vision-lab/practice"
names = os.listdir(folder)
print(names)
print(len(names))

first = names[0]
full_path = os.path.join(folder, first)
print (first)
print (full_path)
print(os.path.getsize(full_path))

total_files = 0
for name in names:
    total_files = total_files + 1

print(f"总共有{total_files}个文件")

py_files = 0
for name in names:
    if name.endswith(".py"):
        py_files = py_files + 1
print(f"总共有{py_files}个.py文件")

total_bytes = 0
for name in names:
    full_path = os.path.join(folder, name)
    total_bytes = total_bytes + os.path.getsize(full_path)
print(f"总共有{total_bytes}字节")
print(f"总共有{total_bytes/1024}KB")
print(f"总共有{total_bytes/1024/1024}MB")
print(f"总共有{total_bytes/1024/1024/1024}GB")
