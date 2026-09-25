import csv

def read_scores(path):
    with open(path,"r",encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  
        rows = []
        for row in reader:
            rows.append([row[0],int(row[1])])
    return rows

def average(rows):
    total = 0
    for row in rows:
        total += row[1]
    return total / len(rows)

def best(rows):
    top_row = rows[0]
    for row in rows:
        if row[1] > top_row[1]:
            top_row = row
    return top_row

def count(rows):
    return len(rows)

def lowest(rows):
    lowest_row = rows[0]
    for row in rows:
        if row[1] < lowest_row[1]:
            lowest_row = row
    return lowest_row

def above_average(rows):
    avg = average(rows)
    above_rows = []
    for row in rows:
        if row[1] > avg:
            above_rows.append(row[0])
    return above_rows

if __name__ == "__main__":
    print(f"我是 score_tools，我的名字是：{__name__}")
    rows = read_scores("D:/AI/zcode files/vision-lab/practice/成绩.csv")
    print(f"试跑结果：平均{average(rows)}，最高{best(rows)}")   
