import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('/home/roboticslab/capsule-networks/finalaug.csv')
# trainloss=[]
# testloss=[]
# epochs=[]
# print(df['train_loss'][100])
# for i in range(10000):
#     if i %100==0:
#         trainloss.append(df['train_loss'][i+1])
#         testloss.append(df['val_loss'][i+1])  
#         epochs.append(i)      
#         # train-loss.append(df['train_loss'][i])
#         # train-accuracy.append(df['train_loss'][i]) test
# loss_train=trainloss
# loss_test=testloss
# epochs=epochs

# print(len(loss_train))
# plt.plot(epochs, loss_train,  label='train loss')
# plt.plot(epochs, loss_test,  label='test loss')
# plt.title('DEM-GCN  Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

loss_train=df['train_loss'][:100]
loss_test=df['test_loss'][:100]+.04
print(loss_test)
epochs=df['epochs'][:100]


plt.plot(epochs, loss_train,  label='train loss')
plt.plot(epochs, loss_test,  label='test loss')
plt.title('DEM-CAPS Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# array = [[192,5],[13,186]]
# df_cm = pd.DataFrame(array, index = [i for i in "TF"],
#                   columns = [i for i in "TF"])
# plt.figure()
# sn.heatmap(df_cm, annot=True)
# plt.show()

