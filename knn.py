import pandas as pd
import numpy as np

df = pd.read_csv("/content/sample_data/mnist_train_small.csv")

df.head()

df.rename(columns = {'6':'label'}, inplace = True)

df.head(1)

len(df.columns)

x = [x for x in range(len(df.columns))]
x[0] = 'Label'
df.columns = x

df.head()

df[560].value_counts()

X_test = pd.read_csv('/content/sample_data/mnist_test.csv')

X_test.head()



x = [x for x in range(len(X_test.columns))]
x[0] = 'Label'
X_test.columns = x

X_test.head(1)

y_test = X_test.pop('Label')

y_train = df.pop('Label')
X_train = df.copy()

# X_train = X_train.to_numpy()
# X_test = X_test.to_numpy()
# y_train = np.array(y_train)
# y_test = np.array(y_test)

len(y_test)

len(y_train)

class knn:
    def __init__(self,X_train,y_train,X_test,y_test,k):
        self.X_train = X_train        
        self.X_test = X_test
        self.y_test = y_test
        self.y_train = y_train
        self.k = k

    def dist(self,x,y):
        if len(x) == len(y):
            sum = 0

            for i in range(0,len(x),1):
                sum += (x[i] - y[i]) ** 2
            return sum ** 0.5

        else:
            return -1

    def test(self):
        
        y_pred = list()

        for index,x_test in self.X_test.iterrows():
            d = dict()

            if True:
                for ind,x_train in self.X_train.iterrows():
                    d[ind] = self.dist(np.array(x_test),np.array(x_train))

                d = {k: v for k, v in sorted(d.items(), key=lambda item: item[1])}

                count = 0

                neighbors = list()

                for i in d:
                    if count < self.k:
                        neighbors.append(self.y_train[i])

                    else:
                        break

                    count += 1

                k_neigh = dict()

                for i in neighbors:
                    if i not in k_neigh:
                        k_neigh[i] = 1
                        continue

                    else:
                        k_neigh[i] += 1

                k_neigh = {k: v for k, v in sorted(k_neigh.items(), key=lambda item: item[1],reverse=True)}
                

                for i in k_neigh:
                    y_pred.append(i)
                    break

                print(k_neigh)
                print(neighbors)
        print(y_pred)

model = knn(X_train,y_train,X_test[:][:10],y_test[:10],5)
model.test()

