import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

import warnings
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve
warnings.filterwarnings("ignore")

data=pd.read_csv('X.csv',header=None)
data=np.array(data)
datay=pd.read_csv('y.csv',header=None)
datay=np.array(datay)
m=data.shape[0]
n=data.shape[1]
means=[]
stds=[]
for i in range(n):
	means.append(np.mean(data[0:m,i]))
	stds.append(np.std(data[0:m,i]))
for i in range(m):
	for j in range(n):
		data[i,j]=(data[i,j]-means[j])/stds[j]

datatest=pd.read_csv('Xt.csv',header=None)
datatest=np.array(datatest)
dataytest=pd.read_csv('yt.csv',header=None)
dataytest=np.array(dataytest)
m=datatest.shape[0]
n=datatest.shape[1]
means=[]
stds=[]
for i in range(n):
	means.append(np.mean(datatest[0:m,i]))
	stds.append(np.std(datatest[0:m,i]))
for i in range(m):
	for j in range(n):
		datatest[i,j]=(datatest[i,j]-means[j])/stds[j]

x=data
y=datay
print("		Training Data")
clf = tree.DecisionTreeClassifier(criterion='gini',)
clf=clf.fit(data, datay)

y_pred=clf.predict(data)
conf=confusion_matrix(datay,y_pred)
print(conf)
sc=accuracy_score(datay,y_pred)
print(classification_report(datay,y_pred))
print("Accuracy:",sc*100)
print()

print("		Testing Data")
y_pred=clf.predict(datatest)
conf=confusion_matrix(dataytest,y_pred)
print(conf)
sc=accuracy_score(dataytest,y_pred)
print(classification_report(dataytest,y_pred))
print("Accuracy:",sc*100)
print()

tree.plot_tree(clf)