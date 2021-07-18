import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import warnings
from sklearn import svm
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
ValuesofC=[0.1,0.5,1,10,25,50]
kf = KFold(n_splits=len(ValuesofC))
fld=1
bestAccuracy=0
bestclf=0
bestTitle=0
bestC=0
for train_index, test_index in kf.split(x):
	C=ValuesofC[fld-1]
	print("    Fold",fld,"with C",C)
	x_train, x_test = x[train_index], x[test_index]
	y_train, y_test = y[train_index], y[test_index]
	models = (svm.SVC(kernel='linear', C=C).fit(x_train,y_train),
	          svm.SVC(kernel='rbf', gamma='scale', C=C).fit(x_train,y_train),
	          svm.SVC(kernel='poly', degree=2, C=C).fit(x_train,y_train))
	modelswoFit = (svm.SVC(kernel='linear', C=C),
	          svm.SVC(kernel='rbf', gamma='scale', C=C),
	          svm.SVC(kernel='poly', degree=2, C=C))
	titles = ('SVC with linear kernel', 'SVC with RBF kernel', 'SVC with polynomial(degree 2) kernel')
	for clf, title, clf1 in zip(models, titles,modelswoFit):
		y_pred=clf.predict(x_test)
		conf=confusion_matrix(y_test,y_pred)
		print(title)
		# print(conf)
		sc=accuracy_score(y_test,y_pred)
		if(sc>bestAccuracy):
			bestclf=clf1
			bestAccuracy=sc
			bestC=C
			bestTitle=title
		print("Accuracy:",sc*100)
		# print(classification_report(y,y_pred))
	print("_____________________________________")
	fld+=1

print("Best is",bestTitle,"with C",bestC)
print()
print("		RESULT ON TRAINING DATA")
# clf=svm.SVC(kernel='rbf',gamma='scale',C=10).fit(data,datay)
clf=bestclf.fit(data,datay)
y_pred=clf.predict(data)
conf=confusion_matrix(datay,y_pred)
print(conf)
sc=accuracy_score(datay,y_pred)
print(classification_report(datay,y_pred))
print("Accuracy:",sc*100)
print()
print("		RESULT ON TESTING DATA")
y_pred=clf.predict(datatest)
conf=confusion_matrix(dataytest,y_pred)
print(conf)
sc=accuracy_score(dataytest,y_pred)
print(classification_report(dataytest,y_pred))
print("Accuracy:",sc*100)

