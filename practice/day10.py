import csv

path = "D:/AI/zcode files/vision-lab/practice/成绩.csv"

with open(path,"r",encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader)
    rows = []
    for row in reader:
        rows.append(row)

print(rows)
print(len(rows))

total = 0
for row in rows:
    total = total + int(row[1])

print(f"总成绩为{total}")
print(f"平均成绩为{total/len(rows)}")

best_score = 0
best_name = ""
for row in rows:
    score = int(row[1])
    if score > best_score:
        best_score = score
        best_name = row[0]

print(f"最高分是{best_score},学生是{best_name}")
