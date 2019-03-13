"""Author: WKirgsn, 2018
Hands-On Training with Self-Normalizing Neural Networks"""

import warnings
warnings.filterwarnings("ignore")
import preprocessing.config as cfg
import numpy as np
import gc
from preprocessing import select_gpu  # choose GPU through import

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow import set_random_seed
from preprocessing.data import LightDataManager
import preprocessing.file_utils as futils
from preprocessing.snn_model_utils import SNNKerasRegressor, build_snn_model

# Ensure reproducibility
SEED = cfg.data_cfg['random_seed']
np.random.seed(SEED)
set_random_seed(SEED)


def main():
    # config
    if cfg.debug_cfg['DEBUG']:
        print('## DEBUG MODE ON ##')

    dm = LightDataManager(cfg.data_cfg['file_path'])
    dm.featurize()

    x_train = dm.tra_df[dm.x_cols]
    y_train = dm.tra_df[dm.y_cols]
    x_val = dm.val_df[dm.x_cols]
    y_val = dm.val_df[dm.y_cols]
    x_tst = dm.tst_df[dm.x_cols]
    y_tst = dm.tst_df[dm.y_cols]

    gc.collect()

    callbacks = [
        EarlyStopping(monitor='val_loss',
                      min_delta=1e-3,
                      patience=5,
                      verbose=1),
        ReduceLROnPlateau(monitor='loss',
                          patience=2), ]


    # start trials
    trial_reports = futils.TrialReports(SEED)
    KerasRegressor_config = {'x_shape': (1, len(dm.x_cols)),
                             'verbose': 1,
                             }

    # add configs from config file (these must match with args of build_fn)
    init_cfg = cfg.keras_cfg['snn_params'].copy()
    init_cfg.pop('batch_size', None)
    KerasRegressor_config.update(init_cfg)
    fit_cfg = {'X': x_train, 'y': y_train,
               'validation_data': (x_val, y_val),
               'callbacks': callbacks,
               }

    for result in trial_reports.conduct(cfg.keras_cfg['n_trials']):
        # create model
        model = SNNKerasRegressor(build_fn=build_snn_model,
                                  **KerasRegressor_config)

        result.history = model.fit(**fit_cfg)
        result.model = model
        result.yhat_te = model.predict(x_tst)
        result.yhat_te = dm.inverse_transform(result.yhat_te)
        result.actual = dm.inverse_transform(y_tst)

        result.score, _ = dm.score(result.actual, result.yhat_te)


if __name__ == '__main__':
    main()
