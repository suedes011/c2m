import pandas as pd
import numpy as np

df = pd.read_csv('/content/drive/MyDrive/ml datasets/tick tak toe/ticktaktoe.csv')

df

df.isnull().sum()

for i in df.columns:
  print(df[i].value_counts())
  print()

df.columns

x_num = pd.get_dummies(df[['V1','V2','V3','V4','V5','V6','V7','V8','V9']],drop_first = True)

df.replace('negative',0,inplace=True)
df.replace('positive',1,inplace=True)

x_num.columns

y = df.iloc[:,9].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_num,y,test_size = 0.33, random_state = 42)

from sklearn.preprocessing import StandardScaler
st = StandardScaler()
x_train = st.fit_transform(x_train)
x_test = st.fit_transform(x_test)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state =  42)
classifier.fit(x_train,y_train)

y_pred= classifier.predict(x_test)
y_pred

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
cm

incorrect_pred = (y_test != y_pred).sum()
incorrect_pred

from sklearn import metrics
from sklearn.metrics import classification_report
metrics.accuracy_score(y_test,y_pred)

print(classification_report(y_test, y_pred))

import matplotlib.pyplot as plt
from sklearn import tree
plt.figure(figsize=(50,30))
tree.plot_tree(classifier,filled=True)
plt.show()

#better visualization
feature = ['V1_o', 'V1_x', 'V2_o', 'V2_x', 'V3_o', 'V3_x', 'V4_o', 'V4_x', 'V5_o','V5_x', 'V6_o', 'V6_x', 'V7_o', 'V7_x', 'V8_o', 'V8_x', 'V9_o', 'V9_x']
class_name = ['Positive','Negative']
import graphviz
dot_data = tree.export_graphviz(classifier, out_file=None,feature_names=feature,class_names=class_name,filled=True)
graph = graphviz.Source(dot_data, format="png") 
graph

from sklearn.metrics import roc_auc_score, roc_curve
dec_tree = tree.DecisionTreeClassifier()
dec_tree.fit(x_train,y_train)
pred = dec_tree.predict_proba(x_test)
roc_score = roc_auc_score(y_test,pred[:,1])
print("ROC SCORE: ",roc_score)
fpr,tpr,threshold = roc_curve(y_test,pred[:,1])
plt.clf()
plt.plot(fpr,tpr)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve')
plt.show()