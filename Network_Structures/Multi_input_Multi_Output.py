import keras
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
import numpy as np

#todo put real data as opposed to dummy data
headline_data = np.array([np.zeros(100)]*1024)
additional_data = np.array([np.zeros(5)]*1024)
labels = np.array([np.zeros(1)]*1024)


# Headline input: meant to receive sequences of 100 integers, between 1 and 10000.
# Note that we can name any layer by passing it a "name" argument.
main_input = Input(shape=(100,), dtype='int32', name='main_input')

# This embedding layer will encode the input sequence
# into a sequence of dense 512-dimensional vectors.
x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)

# A LSTM will transform the vector sequence into a single vector,
# containing information about the entire sequence
lstm_out = LSTM(32)(x)


auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)


auxiliary_input = Input(shape=(5,), name='aux_input')
x = keras.layers.concatenate([lstm_out, auxiliary_input])

# We stack a deep densely-connected network on top
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)

# And finally we add the main logistic regression layer
main_output = Dense(1, activation='sigmoid', name='main_output')(x)

model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])

model.compile(optimizer='rmsprop', loss='binary_crossentropy',
              loss_weights=[1., 0.2])

model.fit([headline_data, additional_data], [labels, labels],
          epochs=50, batch_size=32)