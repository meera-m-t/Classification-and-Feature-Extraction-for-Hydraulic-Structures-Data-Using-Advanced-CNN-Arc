
import numpy as np 
from sklearn.model_selection import train_test_split
data = np.load(r"/media/Data/CNNtest/5Cdata/data5C.npy") # 5 channels data,(num,100,100,5), channel 0-4 is DEM,RGB,NIR
labels = np.load(r"/media/Data/CNNtest/5Cdata/label.npy")
data=data[:,:,:,:1]

data1=[]
k=np.zeros((100,100,1))
k=k.astype('float32')
for i in range (len(data)):
    if np.array_equal(k,data[i]) ==True:
        print(i)
    else:
        data1.append(data[i])
index = [306,417,418,528,529,924,935,1000,1051,1246,1247,1354,1441,1578,1579,1580,1712,1807,1808]
labels = np.delete(labels, index)
print(labels.shape,"&&&&&&&&&&&&")

data1=np.array(data1)
print(data1.shape)
data_train,data_test,labels_train,labels_test = train_test_split(data1,labels,test_size=0.1,random_state=1,stratify=labels)
print(data_train.shape)
print(data_test.shape)
data=data1
