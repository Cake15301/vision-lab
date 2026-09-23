import csv

def read_scores(path):
    with open(path,"r",encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  
        rows = []
        for row in reader:
            rows.append([row[0],int(row[1])])
    return rows

def count(rows):
    return len(rows)

def lowest(rows):
    lowest_row = rows[0]
    for row in rows:
        if row[1] < lowest_row[1]:
            lowest_row = row
    return lowest_row

def average(rows):
    total = 0
    for row in rows:
        total += int(row[1])
    return total / len(rows)

def above_average(rows):
    above_rows = []
    for row in rows:
        if row[1] > average(rows):
            above_rows.append(row[0])
    return above_rows

    