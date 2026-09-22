import csv

path = "D:/AI/zcode files/vision-lab/practice/成绩.csv"

with open(path, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader)
    rows = []
    for row in reader:
        rows.append(row)

low_score = 100
low_name = ""
score90 = 0
total = 0
for row in rows:
    score = int(row[1])
    if score < low_score:
        low_score = score
        low_name = row[0]
    if score >= 90:
        score90 = score90 + 1
    total = total + score

p_score = total/len(rows)
p_name = []
person = 0
for row in rows:
    score = int(row[1])
    if score > p_score:
        p_name.append(row[0])
        person = person + 1

print(f"平均分是{p_score},高于平均分的学生有{person}人,分别是{p_name}")
print(f"90分以上的学生有{score90}人")
print(f"最低分是{low_score},学生是{low_name}")


