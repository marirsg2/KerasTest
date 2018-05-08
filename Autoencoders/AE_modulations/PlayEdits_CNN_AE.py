from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
from keras import regularizers
import numpy as np
from keras.callbacks import TensorBoard
from keras import metrics

train_model = True
model_weights_file = "Play_edits_CNN_ae_weights.kmdl"

original_dim = 784

#prep the data
(x_train,y_train) , (x_test,y_test) = mnist.load_data()
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
x_train = x_train.reshape((len(x_train),28,28,1))
x_test = x_test.reshape((len(x_test),28,28,1))
print (x_train.shape)
print (x_test.shape)


acceptable_numbers = [1,4,7]
def mnist_reward(in_value):
    if in_value in acceptable_numbers:
        return 1
    else:
        return 0
x_train_reward = np.array([mnist_reward(y) for y in y_train])
x_test_reward = np.array([mnist_reward(y) for y in y_test])


# encoding_dim = 32
input_img = Input(shape=(28,28,1), name="main")
input_reward = Input (shape=(1,), name="reward")

x = Conv2D(16,(3,3),activation='relu', padding='same')(input_img)
x = MaxPooling2D((2,2),padding='same')(x)
x = Conv2D(8,(3,3),activation='relu', padding='same')(x)
x = MaxPooling2D((2,2),padding='same')(x)
x = Conv2D(8,(3,3),activation='relu', padding='same')(x)
encoded = MaxPooling2D((2,2),padding='same')(x)

# from 28x28, max pooled thrice with same padding. 28-14-7-4. 7->4 is with same padding
#now invert the process
x = Conv2D(8,(3,3),activation='relu', padding='same')(encoded)
x = UpSampling2D((2,2))(x)
x = Conv2D(8,(3,3),activation='relu', padding='same')(x)
x = UpSampling2D((2,2))(x)
#todo NOTICE there is no padding here, to match the dimensions needed.
x = Conv2D(16,(3,3),activation='relu')(x)
x = UpSampling2D((2,2))(x)
decoded = Conv2D(1,(3,3),activation='sigmoid',padding='same')(x)

#computing the VAE loss
# xent_loss = metrics.binary_crossentropy(input_img,decoded)
xent_loss = K.mean(metrics.binary_crossentropy(input_img,decoded))
reward_based_loss =  input_reward*xent_loss
# reward_based_loss =  xent_loss

#full AE model
autoencoder = Model(inputs = [input_img,input_reward],outputs = [decoded])
autoencoder.add_loss(reward_based_loss)
autoencoder.compile(optimizer='rmsprop')
autoencoder.summary()

#encoder model
encoder = Model(input_img,encoded)
#decoder model


if not train_model:
    autoencoder.load_weights(filepath=model_weights_file)
else:
    #todo try changing the loss function to return a single value as opposed to an array and see if tensorboard works
    autoencoder.fit([x_train,x_train_reward],epochs=10,batch_size= 200,
                    shuffle=True,validation_data=([x_test,x_test_reward],None))
                    # callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

    # autoencoder.fit([x_train,x_train_reward],epochs=10,batch_size=200,
    #             shuffle=True,callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
    autoencoder.save_weights(model_weights_file)

encoded_imgs = encoder.predict(x_test)
decoded_imgs = autoencoder.predict([x_test,x_test_reward])


import matplotlib.pyplot as plt

n=10 #number of images to be displayed
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2,n,i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(True)#just for fun
    #display reconstruction
    ax = plt.subplot(2,n,i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
