from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.datasets import mnist
from keras import regularizers
import numpy as np
from keras.callbacks import TensorBoard
import random as rand

fraction_of_data = 1.0
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


def keep_sample_by_reward(index, reward):
    #introduce noise into each pixel with probability determined noise_factor
    #done by first generating noise value over an array of zeros, and then
    #AVERAGING the noise with the DATA.
    cutoff= np.random.rand()
    if cutoff < reward:
        return index
    else:
        return -1

#=============================

dict_num_reward = {0:0,     1:0,    2:0,    3:0,    4:0,    5:0,    6:0,    7:0,    8:1,  9:0}
# negative_ratio = 0.2
# negative_proportion = 1
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
new_x_train_indices = [keep_sample_by_reward(i, mnist_reward(y_train[i])) for i in range(x_train.shape[0])]
new_x_train_indices  = set(new_x_train_indices)
new_x_train_indices.remove(-1)
new_x_train_indices = list(new_x_train_indices)
new_x_train_indices = new_x_train_indices[:int(len(new_x_train_indices)/1000)*1000]
target_x_train = x_train[new_x_train_indices]
x_train = target_x_train
y_train = y_train[new_x_train_indices]

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
    autoencoder.fit(x_train,target_x_train,epochs=25,batch_size=25,
                    shuffle=True,validation_data=(x_test,x_test),
                    callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
    autoencoder.save_weights(model_weights_file)

encoded_imgs = encoder.predict(x_test)
decoded_imgs = autoencoder.predict(x_test)


#find the indices of two of each class
needed_numbers = [0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9]
# needed_numbers = [0,1,2,3,4,5,6,7,8,9]
target_indices = []
curr_index = rand.randint(100,1000)
for number in needed_numbers:
    while True:
        curr_index += 1
        if y_test[curr_index] == number:
            target_indices.append(curr_index)
            break
    #end while
#end outer for


import matplotlib.pyplot as plt
n=20 #number of images to be displayed
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2,n,i+1)
    plt.imshow(x_test[target_indices[i]].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(True)#just for fun
    #display reconstruction
    ax = plt.subplot(2,n,i+1+n)
    plt.imshow(decoded_imgs[target_indices[i]].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
