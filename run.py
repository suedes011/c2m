from urllib import request

print("MENU")
print("------------------------------------------")
print("1. knn")
print("2. pca")

ch = int(input("Enter your choice : "))
file = ""
if ch == 1:
    file = "knn"
elif ch == 2:
    file = "pca"
else:
    print("wrong option")
    
data = request.urlopen(f"https://raw.githubusercontent.com/suedes011/c2m/main/{file}.py")
print("------------------------------------------\n")
print(data.read().decode('utf-8'))
