import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import warnings
import seaborn as sb 
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

datatest=pd.read_csv('X.csv',header=None)
datatest=np.array(datatest)
dataytest=pd.read_csv('y.csv',header=None)
dataytest=np.array(dataytest)
means=[]
stds=[]
m=datatest.shape[0]
n=datatest.shape[1]
for i in range(n):
	means.append(np.mean(data[0:m,i]))
	stds.append(np.std(data[0:m,i]))
for i in range(m):
	for j in range(n):
		datatest[i,j]=(datatest[i,j]-means[j])/stds[j]
print("L2 Regularisation")
print("    RESULT ON Training DATA")
clf = LogisticRegression(random_state=0, solver='saga',multi_class='multinomial').fit(data, datay)
y_pred=clf.predict(data)
conf=confusion_matrix(datay,y_pred)
print(conf)
print(classification_report(datay,y_pred))
print("Accuracy on Training Set:",clf.score(data,datay)*100,"%")

print("    RESULT ON Testing DATA")
y_pred=clf.predict(datatest)
conf=confusion_matrix(dataytest,y_pred)
print(conf)
print(classification_report(dataytest,y_pred))
print("Accuracy on Testing Set:",clf.score(datatest,dataytest)*100,"%")

print()
print("L1 Regularisation")
print("    RESULT ON Training DATA")
clf = LogisticRegression(penalty='l1',random_state=0, solver='saga',multi_class='multinomial').fit(data, datay)
y_pred=clf.predict(data)
conf=confusion_matrix(datay,y_pred)
print(conf)
print(classification_report(datay,y_pred))
print("Accuracy on Training Set:",clf.score(data,datay)*100,"%")

print("    RESULT ON Testing DATA")
y_pred=clf.predict(datatest)
conf=confusion_matrix(dataytest,y_pred)
print(conf)
print(classification_report(dataytest,y_pred))
print("Accuracy on Testing Set:",clf.score(datatest,dataytest)*100,"%")


print()
print("USING STOCKASTIC GRADIEND DESCENT")
from sklearn.linear_model import SGDClassifier
clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=1000)
clf.fit(data, datay)  
print("L2 Regularisation")
print("    RESULT ON Training DATA")
y_pred=clf.predict(data)
conf=confusion_matrix(datay,y_pred)
print(conf)
print(classification_report(datay,y_pred))
print("Accuracy on Training Set:",clf.score(data,datay)*100,"%")

print("    RESULT ON Testing DATA")
y_pred=clf.predict(datatest)
conf=confusion_matrix(dataytest,y_pred)
print(conf)
print(classification_report(dataytest,y_pred))
print("Accuracy on Testing Set:",clf.score(datatest,dataytest)*100,"%")

clf = SGDClassifier(loss="hinge", penalty="l1", max_iter=1000)
clf.fit(data, datay)
print("L1 Regularisation")
print("    RESULT ON Training DATA")
y_pred=clf.predict(data)
conf=confusion_matrix(datay,y_pred)
print(conf)
print(classification_report(datay,y_pred))
print("Accuracy on Training Set:",clf.score(data,datay)*100,"%")

print("    RESULT ON Testing DATA")
y_pred=clf.predict(datatest)
conf=confusion_matrix(dataytest,y_pred)
print(conf)
print(classification_report(dataytest,y_pred))
print("Accuracy on Testing Set:",clf.score(datatest,dataytest)*100,"%")