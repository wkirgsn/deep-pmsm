"""Author: WKirgsn, 2018
Hyper Parameter Tuning with BayesOpt for Neural Networks (CNN)"""

import preprocessing.config as cfg
import uuid
import argparse
import gc
from preprocessing import select_gpu  # choose GPU through import
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from skopt.callbacks import CheckpointSaver
from skopt import load as skopt_load

from preprocessing.data import LightDataManager
import preprocessing.file_utils as futils
from preprocessing.cnn_model_utils import CNNKerasRegressor, build_cnn_model
from preprocessing.rnn_model_utils import NaNCatcher
from preprocessing.file_utils import TrialReports
from opt.bayes_search import BayesSearchTrials


def main():
    # config
    if cfg.debug_cfg['DEBUG']:
        print('## DEBUG MODE ON ##')
        cfg.keras_cfg['cnn_params']['epochs'] = 1
        cfg.keras_cfg['n_trials'] = 2

    parser = argparse.ArgumentParser(description='Hyper Parameter Tuning CNN')
    parser.add_argument('-i', '--optimizer_uid', required=False,
                        help='The 4-digit uid in hex of the optimizer to load '
                             'if it exists. Will create new optimizer if not '
                             'specified or no checkpoint is found')
    args = parser.parse_args()

    opt_search, optimizer_uid, optimizer_callbacks = \
        BayesSearchTrials.load_checkpoint(args.optimizer_uid)

    dm = LightDataManager(cfg.data_cfg['file_path'])

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
                      baseline=10,  # val_loss must be better than this
                      verbose=1),
        ReduceLROnPlateau(monitor='loss',
                          patience=
                          cfg.keras_cfg['early_stop_patience'] // 3),
        NaNCatcher(),
    ]

    KerasRegressor_config = {'x_shape': (cfg.keras_cfg['window_size'],
                                         len(dm.x_cols)),
                             'verbose': 1,
                             'loss': dm.loss_func
                             }
    # add configs from config file (these must match with args of build_fn)
    KerasRegressor_config.update(cfg.keras_cfg['cnn_params'])

    trial_reports = TrialReports(seed=cfg.data_cfg['random_seed'])

    init_kwargs = {'build_fn': build_cnn_model}
    init_kwargs.update(KerasRegressor_config)

    batch_size = cfg.keras_cfg['cnn_params'].pop('batch_size')
    window_size = cfg.keras_cfg['window_size']

    fit_cfg = {'x': x_train,
               'y': y_train,
               'batch_size': batch_size,
               'validation_data': (x_val, y_val),
               'epochs': cfg.keras_cfg['cnn_params']['epochs'],
               'shuffle': True,
               'callbacks': callbacks,
               'p_id_col': dm.PROFILE_ID_COL,
               'data_cache': trial_reports.data_cache,
               'window_size': window_size
               }

    predict_cfg = {'x': x_tst,
                   'batch_size': batch_size,
                   'p_id_col': dm.PROFILE_ID_COL,
                   'window_size': window_size}
    inverse_transform_cfg = {'df': y_tst,
                             'cols_to_inverse': dm.y_cols}

    search_space = cfg.keras_cfg['hp_skopt_space_cnn']
    opt_search = \
        BayesSearchTrials(CNNKerasRegressor,
                          n_iter=cfg.keras_cfg['hp_tune_params']['n_iter'],
                          search_spaces=search_space,
                          init_params=init_kwargs,
                          fit_params=fit_cfg,
                          predict_params=predict_cfg,
                          inverse_params=inverse_transform_cfg,
                          optimizer_kwargs=
                          {'n_initial_points':
                               cfg.keras_cfg['hp_tune_params'][
                                   'n_init_points']},
                          random_state=cfg.data_cfg['random_seed'],
                          verbose=1,
                          n_seeds=
                          cfg.keras_cfg['hp_tune_params']['seeds_per_trial'],
                          n_jobs=cfg.keras_cfg['hp_tune_params']['n_jobs'],
                          data_manager=dm, checkpoint=opt_search,
                          uid=optimizer_uid,
                          known_evals=cfg.bayes_cfg.get(
                              'hp_skopt_cnn_known_evals', None))
    try:
        opt_search.run(callback=optimizer_callbacks)
    except:
        raise
    finally:
        opt_search.dump_result()


if __name__ == '__main__':
    main()
