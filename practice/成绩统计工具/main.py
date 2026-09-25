import score_tools

path = "D:/AI/zcode files/vision-lab/practice/成绩.csv"

print(f"我是 main，我的名字是：{__name__}")

rows = score_tools.read_scores(path)
print(rows)
print(f"平均成绩为{score_tools.average(rows)}")
top = score_tools.best(rows)
print(f"最高分是{top[1]},学生是{top[0]}")
print(f"一共{score_tools.count(rows)}个学生")
low = score_tools.lowest(rows)
print(f"最低分是{low[1]},学生是{low[0]}")
above = score_tools.above_average(rows)
print(f"高于平均分的学生有{len(above)}个，分别是{above}")
