import os
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import AlphaDropout, GaussianNoise
from keras.layers.core import Dense, Flatten
import keras.optimizers as opts

from keras import regularizers
from keras import initializers
from keras.wrappers.scikit_learn import KerasRegressor

import preprocessing.config as cfg


class SNNKerasRegressor(KerasRegressor):
    def save(self, uid):
        path = os.path.join(cfg.data_cfg['model_dump_path'], uid)
        # self.model.save(path + '.h)  # everything saved
        self.model.save_weights(path + '_weights.h5')
        with open(path + '_arch.json', 'w') as f:
            f.write(self.model.to_json())

    def fit(self, X, y, **kwargs):

        X = X.values.reshape(X.shape[0], 1, X.shape[1])
        kwargs['batch_size'] = cfg.keras_cfg['snn_params']['batch_size']
        val_x, val_y = kwargs.get('validation_data', (None, None))
        if val_x is not None:
            kwargs['validation_data'] = \
                (val_x.values.reshape(val_x.shape[0], 1, val_x.shape[1]), val_y)
        kwargs['epochs'] = cfg.keras_cfg['snn_params']['epochs']
        kwargs['shuffle'] = True

        return super().fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        kwargs['batch_size'] = cfg.keras_cfg['snn_params']['batch_size']
        X = X.values.reshape(X.shape[0], 1, X.shape[1])
        return super().predict(X, **kwargs)


def build_snn_model(x_shape=(100, 1, 10),
                  n_layers=1,
                  n_units=64,
                  kernel_reg=1e-9,
                  activity_reg=1e-9,
                  bias_reg=1e-9,
                  dropout_rate=0.5,
                  optimizer='nadam',
                  lr_rate=1e-5,
                  gauss_noise_std=1e-3,
                  n_gpus=0,):
    """build snn model"""

    opts_map = {'adam': opts.Adam, 'nadam': opts.Nadam,
                'adamax': opts.Adamax, 'sgd': opts.SGD,
                'rmsprop': opts.RMSprop}

    snn_cfg = {
        'units': int(n_units),
        #'batch_input_shape': (batch_size, x_shape[1], x_shape[2]),
        #'batch_size': batch_size,
        'input_shape': x_shape,
        'kernel_regularizer': regularizers.l2(kernel_reg),
        'activity_regularizer': regularizers.l2(activity_reg),
        'bias_regularizer': regularizers.l2(bias_reg),
        'kernel_initializer': initializers.lecun_normal(seed=cfg.data_cfg['random_seed']),
        'activation': 'selu',
    }

    model = Sequential()
    model.add(Dense(**snn_cfg))
    model.add(GaussianNoise(gauss_noise_std))
    model.add(AlphaDropout(dropout_rate))
    if n_layers > 1:
        for i in range(n_layers-1):
            snn_cfg.pop('batch_input_shape', None)
            model.add(Dense(**snn_cfg))
            model.add(GaussianNoise(gauss_noise_std))
            model.add(AlphaDropout(dropout_rate))
    model.add(Flatten())  # todo: WHy lol
    model.add(Dense(len(cfg.data_cfg['Target_param_names'])))

    opt = opts_map[optimizer](lr=lr_rate)
    model.compile(optimizer=opt, loss='mse')
    return model