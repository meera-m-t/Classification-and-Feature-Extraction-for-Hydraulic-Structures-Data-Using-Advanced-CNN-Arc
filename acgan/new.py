import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from skimage import io
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from keras.layers import Input,MaxPooling2D
from keras.models import Model, Sequential,load_model
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.datasets import mnist
from keras.optimizers import Adam
import glob
from keras import backend as K
from keras import initializers
from keras.utils import to_categorical 
from sklearn.model_selection import train_test_split
K.common.set_image_dim_ordering('th')
from sklearn.preprocessing import MinMaxScaler
latent_dim = 100+10
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

path='/media/Data/CNNtest/data1/'   #DEM

def read_img(path):
    cate=[path+x for x in os.listdir(path) if os.path.isdir(path+x)] # get F,T folder
    imgs=[]
    labels=[]
    for idx,folder in enumerate(cate): # idx-> 0:F; 1:T; folder-> F,T
        for im in glob.glob(folder+"/*.png"):
            im = im.replace('\\','/')
#            print('reading the images:%s'%(im))                      
            img=io.imread(im)
            
            # Normalize the dataset-MaxMin
            scaler = MinMaxScaler(feature_range=(-1, 1))
            img = scaler.fit_transform(img)
            
            imgs.append(img)
            labels.append(idx)
    return np.asarray(imgs,np.float32),np.asarray(labels,np.int32)

data,label=read_img(path)
data=np.expand_dims(data,3) # add channel dimension, 4D
X_train,X_test,y_train,y_test = train_test_split(data,label,test_size=0.01,random_state=1,stratify=label)
X_train=X_train.reshape(X_train.shape[0],100,100)
print(X_train.shape)
print(y_train.shape)
print(X_train[0])
X_train = X_train.reshape(X_train.shape[0],1,100,100)
print(X_train.shape)
y_train = to_categorical(y_train)
adam = Adam(lr=0.0002, beta_1=0.5)



generator = Sequential()
# Transforms the input into a 25 × 25 64-channel feature map
generator.add(Dense(64*25*25, input_dim=latent_dim))
generator.add(LeakyReLU(0.2))
#generator.add(BatchNormalization())  
generator.add(Reshape((64, 25, 25)))
generator.add(UpSampling2D(size=(2, 2)))
generator.add(Conv2D(128, kernel_size=(5, 5), padding='same'))
generator.add(LeakyReLU(0.2))
generator.add(UpSampling2D(size=(2, 2)))
generator.add(Conv2D(256, kernel_size=(5, 5), padding='same'))
generator.add(LeakyReLU(0.2))
generator.add(UpSampling2D(size=(1, 1)))
# Produces a 100 × 100 1-channel feature map (shape of a DEM image)
generator.add(Conv2D(1, kernel_size=(5, 5), padding='same', activation='tanh'))
#generator.load_weights('models/dcgan_generator_epoch_1.h5')
print(generator.summary())
generator.compile(loss='binary_crossentropy', optimizer=adam)



# Make Discriminator Model
discriminator = Sequential()
discriminator.add(Conv2D(32, kernel_size=(5, 5), strides=(2, 2), padding='same', 
                         input_shape=(1, 100, 100), kernel_initializer=initializers.RandomNormal(stddev=0.02)))
discriminator.add(LeakyReLU(0.2))
#discriminator.add(BatchNormalization())
discriminator.add(Dropout(0.3))
discriminator.add(Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same', 
                         input_shape=(1, 100, 100), kernel_initializer=initializers.RandomNormal(stddev=0.02)))
generator.add(LeakyReLU(0.2))
#discriminator.add(BatchNormalization())
discriminator.add(Dropout(0.3))
discriminator.add(Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same', 
                         input_shape=(1, 100, 100), kernel_initializer=initializers.RandomNormal(stddev=0.02)))
discriminator.add(LeakyReLU(0.2))
#discriminator.add(BatchNormalization())
discriminator.add(Dropout(0.3))
discriminator.add(Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same', 
                         input_shape=(1, 100, 100), kernel_initializer=initializers.RandomNormal(stddev=0.02)))
discriminator.add(LeakyReLU(0.2))
#discriminator.add(BatchNormalization())
discriminator.add(Dropout(0.3))
discriminator.add(Conv2D(256, kernel_size=(5, 5), strides=(2, 2), padding='same', 
                         input_shape=(1, 100, 100), kernel_initializer=initializers.RandomNormal(stddev=0.02)))
discriminator.add(LeakyReLU(0.2))
#discriminator.add(BatchNormalization())
discriminator.add(Dropout(0.3))
discriminator.add(Flatten())
discriminator.add(Dense(1, activation='sigmoid',name="dis_output"))
#discriminator.load_weights('models/dcgan_discriminator_epoch_1.h5')
print(discriminator.summary())
discriminator.compile(loss='binary_crossentropy', optimizer=adam)




# Make Classifier Model
classifier = Sequential()
classifier.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu',padding='same',
                 input_shape=(1,100,100)))

#classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Conv2D(filters=64, kernel_size=(3,3),activation='relu', padding='same',
                 input_shape=(1,100,100)))
#classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same',
                 input_shape=(1,100,100)))
#classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Conv2D(filters=128, kernel_size=(3,3),activation='relu', padding='same'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Flatten())
classifier.add(Dense(128,activation='relu'))
classifier.add(Dense(2, activation='softmax',name="class_output"))
#classifier.load_weights('models/dcgan_classifier_epoch_1.h5')
print(classifier.summary())
print(classifier.name)
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Creating the Adversarial Network. We need to make the Discriminator weights
# non trainable. This only applies to the GAN model.


discriminator.trainable = False
classifier.trainable = False
ganInput = Input(shape=(latent_dim,))
x = generator(ganInput)
gan = Model(inputs=ganInput, outputs=[discriminator(x),classifier(x)])
losses = {
	discriminator.name: "binary_crossentropy",
	classifier.name: "categorical_crossentropy",
}
lossWeights = {discriminator.name: 1.0, classifier.name: 1.0}
gan.compile(loss=losses, loss_weights=lossWeights, optimizer=adam)

dLosses = []
gLosses = []
cLosses = []

# Plot the loss from each batch
def plotLoss(epoch):
    plt.figure(figsize=(10, 8))
    plt.plot(dLosses, label='Discriminitive loss')
    plt.plot(gLosses, label='Generative loss')
    plt.plot(cLosses, label='Classifier loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('images/dcgan_loss_epoch_%d.png' % epoch)

# Create a wall of generated MNIST images
def plotGeneratedImages(epoch, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, latent_dim-10])
    labels = None
    for i in range(10):
      for j in range(10):
        if labels is None:
          labels = np.array([[int(i==k) for k in range(10)]])
        else:
          labels = np.concatenate((labels,np.array([[int(i==k) for k in range(10)]])),axis=0)
    print(labels.shape)
    noise = np.concatenate((noise,labels),axis=1)
    print(noise.shape)    
    generatedImages = generator.predict(noise)
    print(generatedImages.shape)
    plt.figure(figsize=figsize)
    imgset=np.array(generatedImages)
    np.save("images/dcgan_generated_image.npy",imgset)
	
    for i in range(generatedImages.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generatedImages[i, 0], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('images/dcgan_generated_image_epoch_%d.png' % epoch)

# Save the generator and discriminator networks (and weights) for later use
def saveModels(epoch):
    generator.save('models/dcgan_generator_epoch_%d.h5' % epoch)
    discriminator.save('models/dcgan_discriminator_epoch_%d.h5' % epoch)
    classifier.save('models/dcgan_classifier_epoch_%d.h5' % epoch)

"""## Train our GAN and Plot the Synthetic Image Outputs 
After each consecutive Epoch we can see how synthetic images being improved
"""

epochs = 20
batchSize = 20
batchCount = X_train.shape[0] / batchSize

print('Epochs:', epochs)
print('Batch size:', batchSize)
print('Batches per epoch:', batchCount)

for e in range(1, epochs+1):
    print('-'*15, 'Epoch %d' % e, '-'*15)
    for i in tqdm(range(int(batchCount))):
        # Get a random set of input noise and images
        noise = np.random.normal(0, 1, size=[batchSize, latent_dim])
        random_index = np.random.randint(0, X_train.shape[0], size=batchSize)
        imageBatch = X_train[random_index]
        imgset=np.array(imageBatch)
        np.save("images1/orignal_%d.npy"%i,imgset)

        labels = y_train[random_index]
        imgset=np.array(labels)
        np.save("images2/lables_%d.npy"%i,imgset)
        # Generate fake MNIST images
        generatedImages = generator.predict(noise)
        X = np.concatenate([imageBatch, generatedImages])

        # Labels for generated and real data
        yDis = np.zeros(2*batchSize)
        # One-sided label smoothing
        yDis[:batchSize] = 0.9

        # Train discriminator
        discriminator.trainable = True
        dloss = discriminator.train_on_batch(X, yDis)
        
        # Train Classifier
        classifier.trainable = True
        closs,_ = classifier.train_on_batch(np.concatenate([imageBatch]),labels)
        
        # Train generator
        noise = np.random.normal(0, 1, size=[batchSize, latent_dim-2])
        noise = np.concatenate((noise,labels),axis=1)
        #print(noise.shape)
        yGen = np.ones(batchSize)
        discriminator.trainable = False
        classifier.trainable = False
        gloss = gan.train_on_batch(noise, {discriminator.name:yGen,classifier.name:labels})

    # Store loss of most recent batch from this epoch
    dLosses.append(dloss)
    gLosses.append(gloss)
    cLosses.append(closs)
 
    if e == 1 or e % 1 == 0:
        # Plot losses from every epoch
        plotGeneratedImages(e)
        plotLoss(e)
        saveModels(e)
