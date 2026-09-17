import os


def say_hello(name):
    print(f"你好，{name}!")


say_hello("小明")
say_hello("小红")


def check(number):
    if number % 2 == 0:
        print(f"{number}是偶数")
    else:
        print(f"{number}是奇数")


for number in range(1, 11):
    check(number)


def add_print(a, b):
    print(a + b)


def add_return(a, b):
    return a + b


x = add_print(3, 5)
print("x 里面装的是：", x)
y = add_return(3, 5)
print("y 里面装的是：", y)
print("拿 y 再算一步：", y + 10)


def is_even(number):
    return number % 2 == 0


for number in range(1, 11):
    if is_even(number):
        print(f"{number}是偶数")
    else:
        print(f"{number}是奇数")


def files(folder):
    return len(os.listdir(folder))

n = files("D:/AI/zcode files/vision-lab/practice")
print(f"文件数量是:{n}")