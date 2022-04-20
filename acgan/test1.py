from keras.models import load_model
import numpy as np
latent_dim = 150+10
model=load_model('models/dcgan_generator_epoch_100.h5')
noise = np.random.normal(0, 1, size=[100, latent_dim-2])
labels = None
for i in range(10):
    for j in range(10):

        if labels is None:
          labels = np.array([[int(i==k) for k in range(2)]])
          print(labels)
        else:
          labels = np.concatenate((labels,np.array([[int(i==k) for k in range(2)]])),axis=0)
          print(labels,j)
print(labels.shape)
# noise = np.concatenate((noise,labels),axis=1)
# x=model.predict(noise)



