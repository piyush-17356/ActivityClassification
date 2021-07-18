import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import warnings
from sklearn.manifold import TSNE
import seaborn as sb 
warnings.filterwarnings("ignore")

data=pd.read_csv('X.csv',header=None)
datay=pd.read_csv('y.csv',header=None)

data = (data - data.mean())/data.std()
datay = (datay - datay.mean())/datay.std()
# f,ax = plt.subplots(figsize=(20,20))
# sb.heatmap(data.corr(),annot=True,linewidths=1,fmt='.1f',ax = ax)
# plt.show()
X=data[:100]
y=datay[:100]
X_2d = TSNE(n_components=2).fit_transform(X)
print(X_2d.shape)
# X_2d = X_embedded.fit_transform(X)
target_ids = range(12)
plt.figure(figsize=(6, 5))
# colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
xplot=[]
yplot=[]
print(y)
for j in range(12):
	xplot.append([])
	yplot.append([])
	print(y[j,1])
	# for i in range(len(X_2d)):
		# print(y[i,0])
		# if(y[i][0]==j):
		# 	xplot[len(xplot)-1].append(X_2d[i,0])
		# else:
		# 	yplot[len(yplot)-1].append(X_2d[i,1])

for i in zip(target_ids):
    plt.scatter( X_2d[y == i, 0], X_2d[y == i, 1], label=i)
plt.legend()
plt.show()

