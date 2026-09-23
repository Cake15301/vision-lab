import score_tools_b

path = "D:/AI/zcode files/vision-lab/practice/成绩.csv"

rows = score_tools_b.read_scores(path)
print(f"一共{score_tools_b.count(rows)}个学生")
print(f"最低分是{score_tools_b.lowest(rows)[1]},学生是{score_tools_b.lowest(rows)[0]}")
print(f"高于平均分的学生有{len(score_tools_b.above_average(rows))}个，分别是{score_tools_b.above_average(rows)}")