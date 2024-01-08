import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import Data
data = pd.read_csv("D:\Stroke_Prediction\healthcare-dataset-stroke-data.csv")


#VISUALIZE DATASET
#print(data.plot(kind="box"))
#print(plt.show())
#print(data.info())

'''Splitting Data for Train & Test
x ---train_X,test_X
y ---train_Y, test_Y
'''

X=data.drop('stroke', axis=1)
Y=data['stroke']
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.2,random_state=101)
#print(X_train)
#print(Y_train)
#print(data.describe())
#print(data.info())

from sklearn.preprocessing import StandardScaler
std=StandardScaler()
X_train_std=std.fit_transform(X_train)
X_test_std=std.transform(X_test)
#print(X_train_std)


#TRAINING


#DECISION TREE

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train_std, Y_train)
Y_pred=dt.predict(X_test_std)
from sklearn.metrics import accuracy_score
#ACCURACY OF DECISION TREE
acc_dt=accuracy_score(Y_test, Y_pred)



#LOGISTIC REGRESSION

from sklearn.linear_model import LogisticRegression
lr= LogisticRegression()
lr.fit(X_train_std, Y_train)
Y_pred1=lr.predict(X_test_std)
#print(Y_test)
acc_lr=accuracy_score(Y_test, Y_pred1)



#K-NEAREST NEIGHBOUR

from sklearn.neighbors import KNeighborsClassifier
knn= KNeighborsClassifier()
knn.fit(X_train_std, Y_train)
Y_pred2=knn.predict(X_test_std)
acc_knn=accuracy_score(Y_test, Y_pred2)



#RANDOM FOREST

from sklearn.ensemble import RandomForestClassifier
rf= RandomForestClassifier()
rf.fit(X_train, Y_train)
Y_pred3= rf.predict(X_test)
acc_rf=accuracy_score(Y_test, Y_pred3)



#SUPPORT VECTOR MACHINE

from sklearn.svm import SVC
sv=SVC()
sv.fit(X_train_std, Y_train)
Y_pred4 =sv.predict(X_test_std)
acc_svm=accuracy_score(Y_test, Y_pred4)

#VISUALIZATION
print(plt.bar(['DecisionTree', 'LogisticRegression', 'K-NearestNeighbour', 'RandomForest', 'SVM'],[acc_dt, acc_lr, acc_knn, acc_rf, acc_svm]))
plt.xlabel("Algorithms")
plt.ylabel("Accuracy")
#print(plt.show())
import pickle
import os
scaler_path=os.path.join('D:\Stroke_Prediction','models/scaler.pkl')
with open(scaler_path,'wb') as scaler_file:
    pickle.dump(std,scaler_file)

import joblib
model_path=os.path.join('D:\Stroke_Prediction','models/dt.sav')
joblib.dump(dt,model_path)










