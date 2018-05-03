

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist

batch_size = 100
original_dim = 784
latent_dim = 2
intermediate_dim = 256
epochs = 50
epsilon_std = 1.0

x = Input(batch_shape=(batch_size,original_dim))
h = Dense (intermediate_dim,activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

#--------------------------------------
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0],latent_dim),\
                              mean = 0, stddev=epsilon_std)
    #todo notice that /2 of log variance is sqrt of variance = std_dev
    return z_mean + K.exp(z_log_var/2) * epsilon
#--------------------------------------

# note that "output_shape" isn't necessary with the TensorFlow backend
#todo NOTE that the output shape is already determined by z_mean and log_Var (of size latent dim)
#since Lambda layer is merely applying a matrix function, the output shape CAN be assumed to be that
#of the inputs(I think). If different, then need to specify.
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

#we want some layers to be reused, and so will be separately instantiated
#notice that they are not given the additional argument of the previous layer
decoder_h = Dense(intermediate_dim, activation='relu')#h = hidden
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)#NOW we connect it to the "z" (latent var) layer that sampled from N(0,1) and mean,var
x_decoded_mean = decoder_mean(h_decoded)

#instantiate VAE model
vae = Model(x,x_decoded_mean)

#todo read carefully
#computing the VAE loss
xent_loss = original_dim*metrics.binary_crossentropy(x,x_decoded_mean)
#this is the kl divergence between ? two normal distributions. Easy in closed form,
#and one of the reasons we sample from another normal distribution. Using a parametric distribution
#like normal, can help by getting such closed forms.
kl_loss =  -0.5*K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(xent_loss + kl_loss)

#todo UNSURE if this model structure is the best. I think I've seen it where the KL
#divergence optimizes part of the model, and the reconstruction error alone is used
#to update the full model. So KL loss does not affect the decoder.

vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop')
vae.summary