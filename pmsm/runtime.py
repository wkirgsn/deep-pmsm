import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Flatten, add, Conv1D,\
    Activation, MaxPool1D
from tensorflow.keras import models, optimizers as opts
from tqdm import tqdm
import time

N_TARGETS = 4


def rnn_stator_model():
    n_targets = 3
    n_units = 256

    x = tf.keras.Input(shape=(1, 91))
    x_before = x
    # layer 1
    y = LSTM(units=n_units)(x)
    x_dense = Dense(n_units, activation='relu')(x_before)
    y = add([x_dense, y])

    # layer 2
    y_before = y
    y = LSTM(units=n_units)(y)
    y = add([y_before, y])

    y = Flatten()(y)
    y = Dense(n_targets)(y)
    model = models.Model(outputs=y, inputs=x_before)
    model.compile(optimizer='adam', loss='mse')
    return model


def rnn_rotor_model():
    n_targets = 1
    x = tf.keras.Input(shape=(1, 91))
    x_before = x
    y = LSTM(units=4)(x)
    x_dense = Dense(4, activation='relu')(x_before)
    y = add([x_dense, y])
    y = Flatten()(y)
    y = Dense(n_targets)(y)
    model = models.Model(outputs=y, inputs=x_before)
    model.compile(optimizer='adam', loss='mse')
    return model


def cnn_stator_model():
    # input shape: window_size x len(x_cols)
    n_units = 121
    l_kernel = 6
    w = 32
    x = Input((w, 91))
    # tf.keras knows no causal padding :(
    # layer 1
    y = Conv1D(filters=n_units, kernel_size=l_kernel, activation='relu',
               padding='same')(x)
    # layer 2
    y = Conv1D(filters=n_units, kernel_size=l_kernel, activation='relu',
               padding='same', dilation_rate=2)(y)
    shortcut = Conv1D(filters=n_units, kernel_size=1, dilation_rate=2,
                      padding='same')(x)
    y = add([shortcut, y])

    shortcut = y
    # layer 3
    y = Conv1D(filters=n_units, kernel_size=l_kernel, activation='relu', padding='same')(x)
    # layer 4
    y = Conv1D(filters=n_units, kernel_size=l_kernel, activation='relu',
               padding='same', dilation_rate=4)(y)
    y = add([shortcut, y])

    y = MaxPool1D(pool_size=w)(y)
    y = Flatten()(y)
    y = Dense(units=1)(y)

    model = models.Model(inputs=x, outputs=y)
    model.compile(optimizer=opts.Adam(lr=1e-9), loss='mse')
    return model


def cnn_rotor_model():
    # input shape: window_size x len(x_cols)
    x = Input((33, 91), name='input_62')
    y = Conv1D(filters=126, kernel_size=2, padding='same', activation='relu')(x)

    y = Conv1D(filters=126, kernel_size=2, padding='same',
                      dilation_rate=2, activation='relu')(y)

    shortcut = Conv1D(filters=126, kernel_size=1,
                             dilation_rate=2, padding='same')(x)
    y = add([shortcut, y])

    y = MaxPool1D(pool_size=33)(y)
    y = Flatten()(y)
    y = Dense(units=1, name='dense_62')(y)

    model = models.Model(inputs=x, outputs=y)
    model.compile(optimizer=opts.Adam(lr=1e-9), loss='mse')
    return model


def measure_runtime(model):
    X = np.random.randn(10000, model.input.shape[-2], model.input.shape[-1])
    timings = []
    for row in tqdm(X):
        start = time.clock()
        model.predict(row[np.newaxis, :, :], batch_size=1)
        timings.append(time.clock() - start)
    timings = np.asarray(timings)
    print('Runtime: Mean = {:.6f} ms, '.format(timings.mean()*1000),
          'std = {:.6f} ms'.format(timings.std()*1000),
          'max = {:.6f} ms'.format(timings.max()*1000),
          'min = {:.6f} ms'.format(timings.min()*1000))


def main():
    #model = rnn_stator_model()
    #model = rnn_stator_model()
    model = cnn_stator_model()
    #model = cnn_rotor_model()
    measure_runtime(model)


if __name__=='__main__':
    main()
