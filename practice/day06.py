person = {"name":"刘子健","age":22,"city":"上海"}
print(person["name"])
print(person["age"])
print(person["city"])
print(person)
person["job"] = "学生"
person["age"] = 23
print(person)
for key,value in person.items():
    print(f"{key}:{value}")
names = {0: "person", 1: "car", 2: "dog"}
for key,value in names.items():
    print(f"{key}号是{value}")