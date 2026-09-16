labels=["person","dog","car"]
print(labels)
print(labels[0])
labels.append("bottle")
print(labels)
for x in labels:
    print(f"这是{x}")
print(f"一共检测到{len(labels)}个目标")
labels.append("cat")
print(f"一共检测到{len(labels)}个目标")