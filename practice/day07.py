path = "D:/AI/zcode files/vision-lab/practice/day07_test.txt"

with open(path, "w", encoding="utf-8") as f:
    f.write("第一行\n")
    f.write("第二行\n")

with open(path,"a",encoding="utf-8") as f:
    f.write("第三行\n")
    f.write("第四行\n")

with open(path, "r", encoding="utf-8") as f:
    print(f.read())
    print("---")
    print(f.read())

path = "D:/AI/zcode files/vision-lab/practice/奇偶.txt"

with open(path, "w", encoding="utf-8") as f:
    for x in range(1,11):
        if x % 2 == 0:
            f.write(f"{x}是偶数\n")
        else:
            f.write(f"{x}是奇数\n")

with open(path, "r", encoding="utf-8") as f:
    print(f.read())


def check(number):
    if number % 2 == 0:
        return f"{number}是偶数"
    else:
        return f"{number}是奇数"
    
with open(path, "w", encoding="utf-8") as f:
    for x in range(1,11):
        f.write(check(x) + "\n")
    
with open(path, "r", encoding="utf-8") as f:
    print(f.read())