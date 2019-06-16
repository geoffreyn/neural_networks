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

early_stopping_monitor = EarlyStopping(patience=2)

# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
os.environ["KERAS_BACKEND"] = "tensorflow"

flatx_train = np.reshape(x_train,
                         (-1, x_train.shape[1] * x_train.shape[2], 1, 1))
flaty_train = np.squeeze(flatx_train)
flatx_test = np.reshape(x_test,
                        (-1, x_train.shape[1] * x_train.shape[2], 1, 1))
flaty_test = np.squeeze(flatx_test)


def main(argv):

    if len(argv) > 0:
        skip = int(argv[0])

    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1000, activation='relu', input_shape=(-1, 10, 1, 1)))
    model.add(Dropout(rate=0.5))
    model.add(Dense(x_train.shape[1] * x_train.shape[2], activation='relu'))

    model.compile(loss=keras.losses.mean_squared_error,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    model.summary()

    if os.path.exists('model/autoencoder.hdf'):
        print('Model exists, loading...')
        model.load_weights('model/autoencoder.hdf')
    else:
        skip=False

    if not skip:
        history = AccuracyHistory()

        model.fit(x_train, flaty_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(x_test, flaty_test),
                  callbacks=[history, early_stopping_monitor])
        score = model.evaluate(x_test, flaty_test, verbose=0)
        plt.plot(history.acc)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.show()
        model.save_weights('model/autoencoder.hdf')

    os.makedirs('results/images/', exist_ok=True)

    plt.set_cmap('gray')

    fig, ax = plt.subplots(1, 2)

    img = [ax[0].imshow(np.zeros((28, 28)), vmin=0, vmax=1),
           ax[1].imshow(np.zeros((28, 28)), vmin=0, vmax=1)]

    ax[0].set_title('Original')
    ax[1].set_title('Reconstructed')

    def init():
        img[0].set_data(np.zeros((28, 28)))
        img[1].set_data(np.zeros((28, 28)))

        return (img,)

    def animate(num):
        img[0].set_data(np.squeeze(x_train[num:num+1,:,:,:]))

        img[1].set_data(np.reshape(model.predict(x_train[num:num+1,:,:,:]),
                                (x_train.shape[1], x_train.shape[2])))

        return (img,)

    # call the animator. blit=True means only re-draw the parts that have
    # changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=100, interval=200, blit=False)

    anim.save('results/images/autoencoder.mp4')


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
