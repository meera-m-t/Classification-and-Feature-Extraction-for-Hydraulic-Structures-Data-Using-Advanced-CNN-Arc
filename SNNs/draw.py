import pandas as pd

from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing



df = pd.read_csv('softmax-net12.csv')
print(df.columns)

# loss_2d=df['softmax+AF8-loss+AF8-2D'] 
# loss_4d=df['softmax+AF8-loss+AF8-4D']  
# loss_8d=df['softmax+AF8-loss+AF8-8D'] 
# loss_16d=df['softmax+AF8-loss+AF8-16D'] 

# epochs=df['epochs'][:100]


# plt.plot(epochs, loss_2d,  label='embedding 2D')
# plt.plot(epochs, loss_4d,  label='embedding 4D')
# plt.plot(epochs, loss_8d,  label='embedding 8D')
# plt.plot(epochs, loss_16d,  label='embedding 16D')
# plt.title('Test NLLLoss in The Different Embeddings\' Dimensions')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# df = pd.read_csv('accurcy_online_triple.csv')
# print(df.columns)


loss_2d=df['online+AF8-pair+AF8-accuracy+AF8-4D']

# #/df['online_pair_train_loss_8D'].abs().ma[]x()

loss_4d=df['online+AF8-pair+AF8-train+AF8-accuracy+AF8-4D']
# loss_6d=df['online_pair_loss_8D']
# loss_8d=df['pair_16D']
#/df['online_pair_train_loss_8D'].abs().max()

epochs=df['epochs']


# plt.plot(epochs, loss_2d,  label='train loss')
# plt.plot(epochs, loss_4d,  label='test loss')

# plt.plot(epochs, loss_2d,  label='embedding 2D')
# plt.plot(epochs, loss_4d,  label='embedding 4D')
# plt.plot(epochs, loss_6d,  label='embedding 8D')
# plt.plot(epochs, loss_8d,  label='embedding 16D')

# plt.title('Test OnlineContrastiveLoss in The Different Embeddings\' Dimensions')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()


# plt.title('Accuracy in 2D')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()


# df1 = pd.read_csv('pair.csv')

# print(df1.columns)

# df2 = pd.read_csv('triple.csv')

# print(df2.columns)

# df3 = pd.read_csv('online_pair.csv')

# print(df3.columns)

# df4 = pd.read_csv('online_triple.csv')

# print(df4.columns)


# loss_6=abs(((df['softmax_loss_2D'])-(df['softmax_train_loss_2D'].mean()))/(df['softmax_train_loss_2D'].std()))/10
# loss_7=abs(((df1['pair+AF8-loss+AF8-2D'])-(df1['pair+AF8-train+AF8-loss+AF8-2D'].mean()))/(df1['pair+AF8-train+AF8-loss+AF8-2D'].std()))/10
# loss_8=abs(((df2['triple_loss_4D']) -(df2['triple_train_loss_4D'].mean()))/(df2['triple_train_loss_4D'].std()))/10
# loss_9=abs(((df3['online+AF8-pair+AF8-loss+AF8-8D'])-(df3['online+AF8-pair+AF8-train+AF8-loss+AF8-8D'].mean()))/(df3['online+AF8-pair+AF8-train+AF8-loss+AF8-8D'].std()))/10
# loss_10=abs(((df4['online+AF8-triple+AF8-loss+AF8-2D'])-(df4['online+AF8-triple+AF8-train+AF8-loss+AF8-2D'].mean()))/(df4['online+AF8-triple+AF8-train+AF8-loss+AF8-2D'].std()))/10

# epochs=df['epochs']


# plt.plot(epochs, loss_1,  label='NLLLoss')
# plt.plot(epochs, loss_2,  label='ContrastiveLoss')
# plt.plot(epochs, loss_3,  label='TripletLoss')
# plt.plot(epochs, loss_4,  label='OnlineContrastiveLoss')
# plt.plot(epochs, loss_5,  label='TripletLoss')
# plt.title('Test CNN loss and SNNs\' Losses ')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()




# df = pd.read_csv('softmax.csv')
# print(df.columns)

# loss_2d=df['softmax_train_loss_2D'] 
# loss_4d=df['softmax_loss_2D']  


# epochs=df['epochs'][:100]


plt.plot(epochs, loss_4d,  label='train accuracy')
plt.plot(epochs, loss_2d,  label='test accuracy')
plt.title('Accuracy in 2D')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# df = pd.read_csv('softmax.csv')

# print(df.columns)

# df1 = pd.read_csv('pair.csv')

# print(df1.columns)

# df2 = pd.read_csv('triple.csv')

# print(df2.columns)

# df3 = pd.read_csv('online_pair.csv')

# print(df3.columns)

# df4 = pd.read_csv('online_triple.csv')

# print(df4.columns)


# loss_1=abs(((df['softmax_train_loss_2D'])-(df['softmax_train_loss_2D'].mean()))/(df['softmax_train_loss_2D'].std()))/10
# loss_2=abs(((df1['pair+AF8-train+AF8-loss+AF8-2D'])-(df1['pair+AF8-train+AF8-loss+AF8-2D'].mean()))/(df1['pair+AF8-train+AF8-loss+AF8-2D'].std()))/10
# loss_3=abs(((df2['triple_train_loss_4D']) -(df2['triple_train_loss_4D'].mean()))/(df2['triple_train_loss_4D'].std()))/10
# loss_4=abs(((df3['online+AF8-pair+AF8-train+AF8-loss+AF8-8D'])-(df3['online+AF8-pair+AF8-train+AF8-loss+AF8-8D'].mean()))/(df3['online+AF8-pair+AF8-train+AF8-loss+AF8-8D'].std()))/10
# loss_5=abs(((df4['online+AF8-triple+AF8-train+AF8-loss+AF8-2D'])-(df4['online+AF8-triple+AF8-train+AF8-loss+AF8-2D'].mean()))/(df4['online+AF8-triple+AF8-train+AF8-loss+AF8-2D'].std()))/10

# epochs=df['epochs']


# plt.plot(epochs, loss_1,  label='NLLLoss')
# plt.plot(epochs, loss_2,  label='ContrastiveLoss')
# plt.plot(epochs, loss_3,  label='TripletLoss')
# plt.plot(epochs, loss_4,  label='OnlineContrastiveLoss')
# plt.plot(epochs, loss_5,  label='OnlineTripletLoss')

# # plt.plot(epochs, loss_6,  label='NLLLoss')
# # plt.plot(epochs, loss_7,  label='ContrastiveLoss')
# # plt.plot(epochs, loss_8,  label='TripletLoss')
# # plt.plot(epochs, loss_9,  label='OnlineContrastiveLoss')
# # plt.plot(epochs, loss_10,  label='OnlineTripletLoss')


# plt.title('Train CNN loss and SNNs\' Losses ')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()