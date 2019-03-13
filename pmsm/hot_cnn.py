"""Author: WKirgsn, 2018
Hands-On Training with Recurrent Neural Networks"""

import warnings
warnings.filterwarnings("ignore")
import preprocessing.config as cfg
import gc
import numpy as np
from preprocessing import select_gpu  # choose GPU through import
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow import set_random_seed
from preprocessing.data import LightDataManager
import preprocessing.file_utils as futils
#from preprocessing.model_utils import *
from preprocessing.cnn_model_utils import build_cnn_model, CNNKerasRegressor

# Ensure reproducibility
SEED = cfg.data_cfg['random_seed']
np.random.seed(SEED)
set_random_seed(SEED)


def main():
    # config
    if cfg.debug_cfg['DEBUG']:
        print('## DEBUG MODE ON ##')
        cfg.keras_cfg['cnn_params']['epochs'] = 2
        cfg.keras_cfg['n_trials'] = 2

    dm = LightDataManager(cfg.data_cfg['file_path'])
    dm.featurize()

    # remove batch_size from dict, as it is not needed for build function
    batch_size = cfg.keras_cfg['cnn_params'].pop('batch_size')
    window_size = cfg.keras_cfg['window_size']

    x_train = dm.tra_df[dm.x_cols + [dm.PROFILE_ID_COL]]
    y_train = dm.tra_df[dm.y_cols]
    x_val = dm.val_df[dm.x_cols + [dm.PROFILE_ID_COL]]
    y_val = dm.val_df[dm.y_cols]
    x_tst = dm.tst_df[dm.x_cols + [dm.PROFILE_ID_COL]]
    y_tst = dm.tst_df[dm.y_cols]

    gc.collect()

    callbacks = [
        EarlyStopping(monitor='val_loss',
                      min_delta=1e-3,
                      patience=cfg.keras_cfg['early_stop_patience'],
                      verbose=1),
        ReduceLROnPlateau(monitor='loss',
                          patience=
                          cfg.keras_cfg['early_stop_patience'] // 3),
    ]

    KerasRegressor_config = {'x_shape': (cfg.keras_cfg['window_size'],
                                         len(dm.x_cols)),
                             'verbose': 1,
                             'loss': dm.loss_func
                             }
    # add configs from config file (these must match with args of build_fn)
    KerasRegressor_config.update(cfg.keras_cfg['cnn_params'])


    # start trials
    trial_reports = futils.TrialReports(SEED)
    fit_cfg = {'x': x_train,
               'y': y_train,
               'batch_size': batch_size,
               'window_size': window_size,
               'validation_data': (x_val, y_val),
               'epochs': cfg.keras_cfg['cnn_params']['epochs'],
               'shuffle': True,
               'callbacks': callbacks,
               'p_id_col': dm.PROFILE_ID_COL,
               'data_cache': trial_reports.data_cache, }
    predict_cfg = {'batch_size': batch_size,
                   'window_size': window_size,
                   'p_id_col': dm.PROFILE_ID_COL}

    for result in trial_reports.conduct(cfg.keras_cfg['n_trials']):
        model = CNNKerasRegressor(build_fn=build_cnn_model,
                                  **KerasRegressor_config)
        result.history = model.fit(**fit_cfg)
        result.model = model

        result.yhat_te = model.predict(x_tst, **predict_cfg)
        result.yhat_te = dm.inverse_transform(result.yhat_te)
        result.actual = dm.inverse_transform(y_tst)

        result.score = model.score(result.actual, result.yhat_te,
                                   score_directly=True, batch_size=batch_size,
                                   p_id_col=dm.PROFILE_ID_COL,
                                   loss_func=dm.loss_func)

        # save performance on trainset
        yhat_tr = model.predict(x_train, **predict_cfg)

        result.yhat_tr = (dm.inverse_transform(y_train),
                          dm.inverse_transform(yhat_tr))
    return trial_reports


if __name__ == '__main__':
    trial_reports = main()

