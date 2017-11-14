

import numpy as np
import keras
from keras.models import _clone_sequential_model as sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from matplotlib import pyplot as plt
# print keras.__version__

np.random.seed(123)#for reproducibility

(X_train,y_train),(X_test,y_test) = mnist.load_data()
# print X_train.shape
# plt.imshow(X_train[0])
# plt.show()

#========PREPROCESS DATA FOR KERAS=================
#the depth (number of channels for ConvNets) is 1 for the MNIST data
X_train = X_train.reshape(X_train[0],1,28,28)
X_test = X_test.reshape(X_test[0],1,28,28)
print X_train.shape

print "end"

