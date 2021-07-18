import numpy as np

dat=[0 for k in range(9)]
for i in range(1,10):
    dat[i-1]=np.genfromtxt('PAMAP2_Dataset/Protocol/subject10'+str(i)+'.dat',delimiter=' ',dtype=float)
data=np.concatenate(dat)
means=np.nanmean(data,axis=0)
idxs=np.where(np.isnan(data))
data[idxs]=np.take(means,idxs[1])
data=data[data[:,1]!=0]
np.random.shuffle(data)
temp=[0 for i in range(12)]
k=0
for i in range(1,25):
    dat=data[data[:,1]==i]
    dat=dat[:500]
    if(dat.shape[0]!=0):
        temp[k]=dat
        k=k+1
sdata=np.concatenate(temp)
np.random.shuffle(sdata)
y=sdata[:,1]
id=list(range(54))
id.pop(1)
X=sdata[:,id]
np.savetxt("X.csv",X,delimiter=",")
np.savetxt("y.csv",y,delimiter=",")
