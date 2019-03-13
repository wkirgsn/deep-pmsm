"""Author: WKirgsn, 2018
Hands-On Training with Recurrent Neural Networks"""

import warnings
warnings.filterwarnings("ignore")
from preprocessing import config as cfg
import os
import gc
import numpy as np
from preprocessing import select_gpu  # choose GPU through import
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow import set_random_seed
from preprocessing.data import LightDataManager
import preprocessing.file_utils as futils
from preprocessing.rnn_model_utils import RNNKerasRegressor, \
    build_rnn_model


# Ensure reproducibility
SEED = cfg.data_cfg['random_seed']
np.random.seed(SEED)
set_random_seed(SEED)


def main():
    # config
    if cfg.debug_cfg['DEBUG']:
        print('## DEBUG MODE ON ##')
        cfg.keras_cfg['rnn_params']['epochs'] = 3
        cfg.keras_cfg['n_trials'] = 2

    dm = LightDataManager(cfg.data_cfg['file_path'])
    dm.featurize()

    x_train = dm.tra_df[dm.x_cols + [dm.PROFILE_ID_COL]]
    y_train = dm.tra_df[dm.y_cols + [dm.PROFILE_ID_COL]]
    x_val = dm.val_df[dm.x_cols + [dm.PROFILE_ID_COL]]
    y_val = dm.val_df[dm.y_cols + [dm.PROFILE_ID_COL]]
    x_tst = dm.tst_df[dm.x_cols + [dm.PROFILE_ID_COL]]
    y_tst = dm.tst_df[dm.y_cols]

    gc.collect()

    callbacks = [
        EarlyStopping(monitor='val_loss',
                      min_delta=1e-3,
                      patience=cfg.keras_cfg['early_stop_patience'],
                      verbose=1),
        ReduceLROnPlateau(monitor='loss',
                          patience=cfg.keras_cfg['early_stop_patience'] // 3),]
    # here: batch size = num train profiles * downsample_rate
    batch_size = dm.get_batch_size()
    cfg.keras_cfg['rnn_params']['batch_size'] = batch_size

    KerasRegressor_config = {'x_shape': (batch_size,
                                         cfg.keras_cfg['tbptt_len'],
                                         len(dm.x_cols)),
                             'batch_size': batch_size,
                             'verbose': 1,
                             'n_gpus': len(futils.get_available_gpus()),
                             'loss': dm.loss_func
                             }
    # add configs from config file (these must match with args of build_fn)
    KerasRegressor_config.update(cfg.keras_cfg['rnn_params'])

    # start trials
    trial_reports = futils.TrialReports(SEED)
    fit_cfg = {'x': x_train,
               'y': y_train,
               'batch_size': batch_size,
               'validation_data': (x_val, y_val),
               'epochs': cfg.keras_cfg['rnn_params']['epochs'],
               'shuffle': False,
               'callbacks': callbacks,
               'p_id_col': dm.PROFILE_ID_COL,
               'data_cache': trial_reports.data_cache,
               'downsample_rate': cfg.data_cfg['downsample_rate'],
               'tbptt_len': cfg.keras_cfg['tbptt_len']
               }
    predict_cfg = {'batch_size': batch_size,
                   'p_id_col': dm.PROFILE_ID_COL,
                   'downsample_rate': cfg.data_cfg['downsample_rate'],
                   'tbptt_len': cfg.keras_cfg['tbptt_len']}

    for result in trial_reports.conduct(cfg.keras_cfg['n_trials']):
        model = RNNKerasRegressor(build_fn=build_rnn_model,
                                  **KerasRegressor_config)

        result.history = model.fit(**fit_cfg)
        model.reset_states()
        result.model = model

        result.yhat_te = model.predict(x_tst, **predict_cfg)
        result.yhat_te = dm.inverse_transform(result.yhat_te)
        result.actual = dm.inverse_transform(y_tst)

        result.score = model.score(result.actual, result.yhat_te,
                                   score_directly=True,)

        # save performance on trainset
        yhat_tr = model.predict(x_train, **predict_cfg)
        result.yhat_tr = (dm.inverse_transform(y_train),
                          dm.inverse_transform(yhat_tr))
    return trial_reports


if __name__ == '__main__':
    main()
