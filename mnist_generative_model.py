#!/usr/local/bin python3
import os
import sys

import keras
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.datasets import mnist
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from matplotlib import animation
import matplotlib.pylab as plt
import numpy as np

from mnist_cnn import (x_train, x_test, y_train, y_test, input_shape,
                       batch_size, epochs, AccuracyHistory)
from mnist_autoencoder import (flatx_train, flaty_train,
	                           flatx_test, flaty_test, get_model)


model = get_model(skip=True)

# Load the full model from the mnist_autoencoder and use the nx10 dense layer as an input layer
model_trunc = Sequential()

model_trunc.add(Flatten(input_shape=(10,1,1)))

layers = [l for l in model.layers]
for id_, layer in enumerate(layers[4:]):
    layer.trainable = False
    model_trunc.add(layer)

model_trunc.summary()

model_trunc.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

y_train_extend = np.reshape(y_train, (60000, 10, 1, 1))
y_test_extend = np.reshape(y_test, (10000, 10, 1, 1))

plt.set_cmap('gray')

rows, cols = (4, 3)
fig, ax = plt.subplots(rows, cols)

for num in range(10):
    z = [0]*10
    z[num] = 1
    ax[num // cols][num % cols].imshow(np.reshape(model_trunc.predict(np.reshape(z, (1, 10, 1, 1))), (28, 28)))
    ax[num // cols][num % cols].set_title('#{}'.format(num))

fig.savefig('results/images/generative_model_integers.png')
