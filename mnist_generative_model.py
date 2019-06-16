#!/usr/local/bin python3
import os
import sys

import keras
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from matplotlib import animation
import matplotlib.pylab as plt
import numpy as np

from mnist_autoencoder import (flatx_train, flaty_train,
                               flatx_test, flaty_test, get_model)
from mnist_cnn import (x_train, x_test, y_train, y_test, input_shape,
                       batch_size, epochs, AccuracyHistory)

early_stopping_monitor = EarlyStopping(patience=3)

def main(argv):
    if len(argv) > 0:
        transfer_learning = int(argv[0])
    else:
        transfer_learning = 1

    model = get_model(skip=True)

    # Load the full model from the mnist_autoencoder and use the (n x 10)
    # dense layer as an input layer
    model_trunc = Sequential()

    model_trunc.add(Flatten(input_shape=(10, 1, 1)))

    layers = [l for l in model.layers]
    for id_, layer in enumerate(layers[4:]):
        if transfer_learning:
            layer.trainable = False
        model_trunc.add(layer)

    model_trunc.summary()

    y_train_extend = np.reshape(y_train, (60000, 10, 1, 1))
    y_test_extend = np.reshape(y_test, (10000, 10, 1, 1))

    if not transfer_learning:
        try:
            model_trunc.load_weights('model/autoencoder_truncated.hdf')
            print('Loaded exiting model...')
        except (OSError, ValueError):
            model_trunc.compile(loss=keras.losses.mean_squared_logarithmic_error,
                          optimizer=keras.optimizers.Adam(),
                          metrics=['accuracy'])

            model_trunc.fit(y_train_extend, flatx_train[:,:,0,0],
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            validation_data=(y_test_extend, flatx_test[:,:,0,0]),
                            callbacks=[early_stopping_monitor])

            model_trunc.save_weights('model/autoencoder_truncated.hdf')

    plt.set_cmap('gray')

    rows, cols = (4, 3)
    fig = plt.figure(figsize=(12, 12))

    for num in range(rows*cols):
        ax = fig.add_subplot(rows, cols, num+1)

        if not transfer_learning:
            z = [0]*10
            z[num] = 1
        else:
            z = np.random.rand(10)

        z = z / np.sum(z)

        ax.imshow(np.reshape(model_trunc.predict(np.reshape(z,
                                                            (1, 10, 1, 1))),
                                                            (28, 28)))

        ax.set_title('#{}'.format(num))

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        if not transfer_learning:
            if num >= 9:
                break
    
    fig.savefig('results/images/generative_model_integers_{}.png'.format(transfer_learning))

    # Generate number transition image
    plt.set_cmap('gray')

    fig, ax = plt.subplots()

    img = ax.imshow(np.zeros((28, 28)), vmin=0, vmax=1)

    num1 = np.random.randint(10)
    num2 = num1
    while (num2 == num1):
        num2 = np.random.randint(10)

    ttl = ax.text(.4, 1.05, '{} -> {}: {}%'.format(num1, num2, 0), transform = ax.transAxes, va='center')

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    def init():
        ttl.set_text('')
        img.set_data(np.zeros((28, 28)))

        return (img,)

    def animate(factor):
        ttl.set_text('{} -> {}: {}%'.format(num2, num1, factor))

        z = [0]*10
        z[num1] = factor / 100
        z[num2] = 1 - factor / 100

        result = np.reshape(model_trunc.predict(np.reshape(z,
                                                            (1, 10, 1, 1))),
                                                            (28, 28))
        img.set_data(result)

        return (img,)

    # call the animator. blit=True means only re-draw the parts that have
    # changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=100, interval=50, blit=False)

    anim.save('results/images/generative_transformation.mp4')
    anim.save('results/images/generative_transformation.gif', writer='imagemagick', fps=2)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
