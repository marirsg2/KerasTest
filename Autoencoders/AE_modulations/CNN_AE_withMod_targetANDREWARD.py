from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.datasets import mnist
from keras import regularizers
import numpy as np
from keras.callbacks import TensorBoard
import random as rand

fraction_of_data = 0.25
#prep the data
(x_train,y_train) , (x_test,y_test) = mnist.load_data()
x_train = x_train[0:int(len(x_train)*fraction_of_data)]
y_train = y_train[0:int(len(y_train)*fraction_of_data)]
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
x_train = x_train.reshape((len(x_train),28,28,1))
x_test = x_test.reshape((len(x_test),28,28,1))
print (x_train.shape)
print (x_test.shape)

train_model = True
model_weights_file = "CNN_ae_weights_MODtarget.kmdl"

acceptable_numbers = [6]
def add_noise(index, noise_factor):
    #introduce noise into each pixel with probability determined noise_factor
    #done by first generating noise value over an array of zeros, and then
    #AVERAGING the noise with the DATA.
    noise_mask = np.random.rand(28, 28, 1)
    noise_mask = np.less(noise_mask,noise_factor) #so if the noise factor was 0.4, then only those nodes where value
                                                    #is less than 0.4 will be 1
    noise_layer = np.random.rand(28, 28, 1) #THIS is the actual noise value. DIFFERENT from the one used to generate mask
    return x_train[index]*(1-noise_mask) + noise_layer*noise_mask
#=============================

dict_num_reward = {0:0.5,1:0,2:0,3:0,4:0,5:0,6:0.8,7:0,8:1,9:0,}
negative_ratio = 0.2
negative_proportion = 1
def mnist_reward(in_value):
    # for pure dict values
    return dict_num_reward[in_value]
    #for random negative values
    # value = dict_num_reward[in_value]
    # rand_val = rand.uniform(0,1)
    # if rand_val < negative_ratio:
    #     return value*-1*negative_proportion
    # else:
    #     return value


# 1-mnist_reward = noise factor, and tells us how much noise to generate
new_x_train = [add_noise(i, 1 - mnist_reward(y_train[i])) for i in range(x_train.shape[0])]
target_x_train = np.array(new_x_train)

# encoding_dim = 32
input_img = Input(shape=(28,28,1))

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


#full AE model
autoencoder = Model(input_img,decoded)
autoencoder.compile(optimizer='adadelta', loss = 'binary_crossentropy')
# autoencoder.compile(optimizer='adadelta', loss = 'mse')
#encoder model
encoder = Model (input_img,encoded)
#decoder model

if not train_model:
    autoencoder.load_weights(filepath=model_weights_file)
else:
    autoencoder.fit(x_train,target_x_train,epochs=3,batch_size=32,
                    shuffle=True,validation_data=(x_test,x_test),
                    callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
    autoencoder.save_weights(model_weights_file)

encoded_imgs = encoder.predict(x_train)
decoded_imgs = autoencoder.predict(x_train)
import matplotlib.pyplot as plt
n=20 #number of images to be displayed
plt.figure(figsize=(20,4))
for i in range(n):
    image_index = i*10
    ax = plt.subplot(2,n,i+1)
    plt.imshow(x_train[image_index ].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(True)#just for fun
    #display reconstruction
    ax = plt.subplot(2,n,i+1+n)
    plt.imshow(decoded_imgs[image_index ].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
