from urllib import request

print("MENU")
print("------------------------------------------")
print("1. knn")
print("2. pca")
print("3. decision tree")

ch = int(input("Enter your choice : "))
file = ""
if ch == 1:
    file = "knn"
elif ch == 2:
    file = "pca"
elif ch == 3:
    file = "tic_tac_toe_decisiontree"
else:
    print("wrong option")
    
data = request.urlopen(f"https://raw.githubusercontent.com/suedes011/c2m/main/{file}.py")
print("------------------------------------------\n")
print(data.read().decode('utf-8'))
