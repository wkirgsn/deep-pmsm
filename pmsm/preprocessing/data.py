"""
Author: Kirgsn, 2018, https://www.kaggle.com/wkirgsn
"""
from abc import ABC, abstractmethod
import gc
import numpy as np
import pandas as pd
import multiprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler,\
    PolynomialFeatures
from sklearn.metrics import mean_squared_error as sklearn_mse
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline, make_union
from sklearn.base import BaseEstimator, TransformerMixin
from keras.losses import mean_squared_error, mean_squared_logarithmic_error
import keras.backend as K
from sklearn.model_selection import PredefinedSplit

import preprocessing.config as cfg
from preprocessing.file_utils import measure_time


class ColumnManager:
    """Class to keep track of the current input columns and for general
    economy of features"""
    def __init__(self, df, white_list=[]):
        self.original = list(df.columns)
        self._x, self._y = None, cfg.data_cfg['Target_param_names']
        self.white_list = white_list
        self.update(df)

    @property
    def x_cols(self):
        return self._x

    @x_cols.setter
    def x_cols(self, value):
        self._x = value

    @property
    def y_cols(self):
        return self._y

    def update(self, df):
        x_cols = []
        for col in df.columns:
            for p in cfg.data_cfg['Input_param_names']:
                if p in col:
                    x_cols.append(col)
                    break
            else:
                # col hasn't matched the pattern, check whitelist
                if col in self.white_list:
                    x_cols.append(col)
        self.x_cols = x_cols


class DataManagerScorer:
    """For sklearn functions that requirer a scorer. With this class we can
    un-standardize according to our datamanager and get the unnormalized
    score"""
    __name__ = 'data_manager_loss'  # for tpot

    def __init__(self, datamanager, force_numpy=True, force_ycol=None):
        self.dm = datamanager
        self.force_numpy = force_numpy
        self.force_ycol = force_ycol

    def __call__(self, estimator, X, y):
        """Score function"""
        if self.force_ycol is not None:
            y_col = self.force_ycol
        elif isinstance(y, pd.Series):
            y_col = y.name
            y_test = y.values
        elif isinstance(y, pd.DataFrame):
            y_col = y.columns
            y_test = y.values
        else:
            # user hasnt specified target and y_test is numpy matrix so
            # assume targets from config file
            y_col = cfg.data_cfg['Target_param_names']
        y_hat = estimator.predict(X)
        y_hat = self.dm.inverse_transform(y_hat, cols_to_inverse=y_col)
        y_test = self.dm.inverse_transform(y_test, cols_to_inverse=y_col)
        score = np.average((y_test - y_hat) ** 2, axis=0)
        if self.force_numpy and hasattr(score, 'values'):
            score = score.values
        return score


class DataManager(ABC):
    """Abstract dataset managing class"""

    PROFILE_ID_COL = 'profile_id'
    START_OF_PROFILE_COL = 'p_start'
    loss_funcs = {'mse': mean_squared_error,
                  'msle': mean_squared_logarithmic_error}

    def __init__(self, path, create_holdout=True):
        gc.enable()
        conversion_table = {col: np.float32 for col in cfg.data_cfg[
            'Input_param_names']+cfg.data_cfg['Target_param_names']}
        conversion_table.update({self.PROFILE_ID_COL: np.uint8})
        # original data
        self.dataset = pd.read_csv(path, dtype=conversion_table)
        # When using CV, do not create a hold out
        self.has_hold_out = create_holdout
        self.loss_func = self.loss_funcs[cfg.data_cfg['loss']]

        # downsample
        # self.dataset = self.dataset.iloc[::2, :]

        # Drop profiles
        if cfg.data_cfg['drop'] is not None:
            assert isinstance(cfg.data_cfg['drop'], list), \
                'please specify profiles to drop as list'
            drop_p_l = [int(p) for p in cfg.data_cfg['drop']]
            self.dataset.drop(index=self.dataset[self.dataset[
                self.PROFILE_ID_COL].isin(drop_p_l)].index, inplace=True)

        self._x_cols, self._y_cols = cfg.data_cfg['Input_param_names'], \
                                     cfg.data_cfg['Target_param_names']
        self.original_profiles = self.dataset[self.PROFILE_ID_COL].unique()\
            .tolist()

        # drop columns
        self.dataset = self.dataset.loc[:, self._x_cols + self._y_cols + [
            self.PROFILE_ID_COL]]

        if cfg.debug_cfg['DEBUG']:
            s = ~self.dataset.loc[:, self.PROFILE_ID_COL].duplicated()

            self.dataset = \
                pd.concat(
                    [self.dataset
                         .iloc[s_start:s_start + cfg.debug_cfg['n_debug'], :]
                     for s_start in list(s[s].index)],
                    axis=0)

        self.df = self.dataset.copy()

    @property
    def x_cols(self):
        return self._x_cols

    @x_cols.setter
    def x_cols(self, value):
        self._x_cols = value

    @property
    def y_cols(self):
        return self._y_cols

    @y_cols.setter
    def y_cols(self, value):
        self._y_cols = value

    @property
    def profiles_df(self):
        """list of dfs per profile"""
        unique_profile_ids = self.df[self.PROFILE_ID_COL].unique().tolist()
        return [self.df.loc[self.df[self.PROFILE_ID_COL] == int(p), :]
                    .reset_index(drop=True) for p in unique_profile_ids]

    @property
    def tra_df(self):
        testsets = cfg.data_cfg['testset']
        valsets = cfg.data_cfg['valset']
        profiles_to_exclude = \
            testsets + valsets if self.has_hold_out else testsets
        profiles_to_include = [p for p in self.original_profiles
                               if str(p) not in profiles_to_exclude]
        all_profiles = self.df[self.PROFILE_ID_COL].unique().tolist()
        profiles_with_noise = []
        for p in profiles_to_include:
            for all_p in all_profiles:
                if str(all_p).endswith(str(p)):
                    profiles_with_noise.append(all_p)

        return self.df.loc[self.df[self.PROFILE_ID_COL].isin(
            profiles_with_noise), :].reset_index(drop=True)

    @property
    def val_df(self):
        sub_df = \
            self.df.loc[
                self.df[self.PROFILE_ID_COL].isin(cfg.data_cfg['valset']), :]
        return sub_df.reset_index(drop=True)

    @property
    def tst_df(self):
        sub_df = \
            self.df.loc[
                 self.df[self.PROFILE_ID_COL].isin(cfg.data_cfg['testset']), :]
        return sub_df.reset_index(drop=True)

    @property
    def actual(self):
        sub_df = \
            self.dataset.loc[
            self.dataset[self.PROFILE_ID_COL].isin(cfg.data_cfg['testset']),
            cfg.data_cfg['Target_param_names'] + [self.PROFILE_ID_COL]]
        return sub_df.reset_index(drop=True)

    @abstractmethod
    def get_featurized_sets(self):
        pass

    @abstractmethod
    def inverse_transform(self, _yhat):
        pass

    def score(self, y_hat, y_true=None):
        """Prints score by comparing given y_hat with dataset's target,
        which is in the testset. Returns the actual target data as well"""
        if y_true is not None:
            act = y_true
        else:
            # compare with actual
            target_col = self.y_cols if y_hat.shape[1] > 1 else [self.y_cols[-1]]
            act = pd.concat(self.actual, axis=0, ignore_index=True)[target_col]
            assert set(act.columns.tolist()) == set(y_hat.columns.tolist()), 'ping!'
        score = np.mean(K.eval(self.loss_func(act.values, y_hat.values)))
        print('{:}: {:.6} K²'.format(cfg.data_cfg['loss'], score))
        return score, act

    def get_p_id(self, _df):
        """Get Profile ID of given dataframe. Raises error, if there are
        more than one profile id"""
        p_ids = _df[self.PROFILE_ID_COL].unique().tolist()
        assert len(p_ids) == 1, 'More than one profile given in get_p_id()!'
        return p_ids[0]

    def reset(self):
        del self.df
        self.df = self.dataset.copy()
        self._x_cols, self._y_cols = cfg.data_cfg['Input_param_names'], \
                                     cfg.data_cfg['Target_param_names']
        self.original_profiles = self.dataset[self.PROFILE_ID_COL].unique() \
            .tolist()
        gc.collect()


class LightDataManager(DataManager):
    """Lightweight data managing without scikit pipelines"""

    def __init__(self, path, has_holdout=True, create_rolling_diffs=False,
                 create_lag_feats=False, standardize=True,
                 create_polynomials=False):
        super().__init__(path=path, create_holdout=has_holdout)
        self.downsample_rate = cfg.data_cfg['downsample_rate']
        self.rolling_lookback = cfg.data_cfg['rolling_lookback']
        self.scaler = StandardScaler()
        self.standardize = standardize
        #self.scaler = QuantileTransformer(output_distribution='normal')
        #self.scaler = RobustScaler(with_centering=True, with_scaling=True)
        self.create_lag_feats = create_lag_feats
        self.create_rolling_diffs = create_rolling_diffs
        self.polynoms = create_polynomials

    def reset(self):
        super().reset()
        self.scaler = StandardScaler()

    def soft_reset(self):
        raise NotImplementedError()

    def get_scorer(self, force_numpy=True):
        return DataManagerScorer(self, force_numpy)

    def inverse_transform(self, df, cols_to_inverse=None):
        """If cols_to_inverse == None then assume target columns"""

        if isinstance(df, pd.DataFrame):
            if self.PROFILE_ID_COL in df.columns:
                df = df.drop(self.PROFILE_ID_COL, axis=1)
            df = df.values

        if len(df.shape) == 1:  # make at least 2 dim
            df = df.reshape([-1, 1])

        float_cols = self.x_cols + self.y_cols
        # df is numpy matrix
        if cols_to_inverse is None:
            # df contains targets only
            assert df.shape[1] == len(self.y_cols), 'target mismatch'
            cols_to_inverse = self.y_cols

        dummy_ar = np.zeros((df.shape[0], len(float_cols)))
        idcs = [i for i, c in enumerate(float_cols) if c in cols_to_inverse]
        dummy_ar[:, idcs] = df

        # inverse scaling
        if self.standardize:
            dummy_ar = self.scaler.inverse_transform(dummy_ar)

        inversed = pd.DataFrame(dummy_ar,
                                columns=float_cols)[cols_to_inverse]

        return inversed

    @measure_time
    def featurize(self):
        print('build dataset ..')
        # extra features
        if set(['i_d', 'i_q', 'u_d', 'u_q']).issubset(
                set(self.df.columns.tolist())):
            extra_feats = {'i_s': lambda x: np.sqrt(x['i_d']**2 + x['i_q']**2),
                           'u_s': lambda x: np.sqrt(x['u_d']**2 + x['u_q']**2),
                           'P_el': lambda x: x['i_s']*x['u_s']}
            self.df = self.df.assign(**extra_feats)
        self.x_cols = [x for x in self.df.columns.tolist() if x not
                       in self.y_cols + [self.PROFILE_ID_COL]]

        with multiprocessing.Pool(2) as pool:
            p_df_list = self.profiles_df

            p_df_dict = {str(int(self.get_p_id(df))): df for df in p_df_list}
            if self.create_lag_feats:
                lag_feats = pool.map(self._dig_into_lag_features, p_df_list)
                assert all(len(i)==1 for i in lag_feats), 'ping!'
                lag_feats = {list(i.keys())[0]: list(i.values())[0] for i in
                             lag_feats}

            rolling_feats = pool.map(self._dig_into_rolling_features, p_df_list)
            assert all(len(i) == 1 for i in rolling_feats), 'ping!'
            rolling_feats = {list(i.keys())[0]: list(i.values())[0] for i in
                             rolling_feats}

            to_merge = [p_df_dict, rolling_feats]
            if self.create_lag_feats:
                to_merge.append(lag_feats)
            # merge features together
            merged_p_df_list = \
                [pd.concat([m[k] for m in to_merge], axis=1) for k in
                 sorted(p_df_dict.keys(), key=int)]
            # drop NaNs, add index as time variable
            merged_p_df_list = [df.dropna().rename_axis('time')
                                  .reset_index(drop=False) for df in
                                merged_p_df_list]

            # merge row-wise the different profiles
            self.df = pd.concat(merged_p_df_list, axis=0, ignore_index=True)

            self.x_cols = [x for x in self.df.columns.tolist()
                           if x not in cfg.data_cfg['db_target_scheme'] +
                           [self.PROFILE_ID_COL]]

            # polynomials
            if self.polynoms:
                preserved_df = self.df.loc[:, self.y_cols +
                                              [self.PROFILE_ID_COL]]
                poly = PolynomialFeatures()
                polynomials_arr = poly.fit_transform(self.df[self.x_cols])
                self.x_cols = poly.get_feature_names(input_features=self.x_cols)
                poly_df = pd.DataFrame(polynomials_arr, columns=self.x_cols)
                self.df = pd.concat([poly_df, preserved_df], axis=1)

            float_cols = self.x_cols + self.y_cols
            # standardize
            if self.standardize:
                p_ids = self.df.loc[:, self.PROFILE_ID_COL]
                self.scaler.fit(self.tra_df[float_cols].astype(float))

                self.df = \
                    pd.DataFrame(self.scaler
                                 .transform(self.df[float_cols]),
                                 columns=float_cols).astype(np.float32)\
                    .assign(**{self.PROFILE_ID_COL: p_ids.values})

        gc.collect()
        print(self.df.memory_usage().sum() / 1024**2, 'MB')
        return self

    def __dig_into_lag_features(self, df):
        # Here, lookback is always equal to last seen observation (lagX)
        lookback = self.downsample_rate
        n_lookback = 1
        dfs = []
        for lback in range(lookback, n_lookback*lookback + 1, lookback):
            # e.g. lback € [32,64,96,128], for lookback=32 and n_lookback=4
            lag_feats = [df.shift(lback).astype(np.float32)
                           .fillna(df.iloc[0, :])
                           .add_suffix(f'_lag_{lback}'),
                         df.diff(periods=lback).astype(np.float32)
                           .fillna(df.iloc[0, :])
                           .add_suffix(f'_lag_{lback}_diff')]

            lag_feats += [abs(lag_feats[1]).astype(np.float32)
                              .add_suffix('_abs'),
                          pd.DataFrame(df.values + lag_feats[0].values,
                                       columns=df.columns)
                              .add_suffix(f'_sum')]

            dfs.append(pd.concat(lag_feats, axis=1))
        ret = pd.concat(dfs, axis=1)
        return ret

    def _dig_into_lag_features(self, _df):
        return {str(int(self.get_p_id(_df))):
                     self.__dig_into_lag_features(_df.loc[:, self.x_cols])}

    def __dig_into_rolling_features(self, df):

        lookback = self.rolling_lookback
        if not isinstance(lookback, list):
            lookback = [lookback]

        # get max lookback
        max_lookback = max(lookback)
        # prepad default values until max lookback in order to get unbiased
        # rolling lookback feature during first observations
        dummy = np.zeros((max_lookback, len(df.columns)))
        dummy = pd.DataFrame(dummy, columns=df.columns)

        temp_quants_first_values_d = \
            {lbl: df[lbl].iloc[0] for lbl in ['ambient', 'coolant'] if lbl in
             df.columns}

        # prepad
        df = pd.concat([dummy, df], axis=0, ignore_index=True)
        for col, first_val in temp_quants_first_values_d.items():
            if col in df.columns:
                df.loc[:max_lookback, col] = first_val
        ew_mean = pd.concat(
            [df.ewm(span=lb).mean().astype(np.float32)
                 .rename(columns=lambda c: c+'_ew_rolling_mean_'+str(lb))
             for lb in lookback], axis=1)
        ew_std = pd.concat(
            [df.ewm(span=lb).std().astype(np.float32)
                 .rename(columns=lambda c: c+'_ew_rolling_std_'+str(lb))
             for lb in lookback], axis=1)
        concat_l = [ew_mean,
                    ew_std,
                    ]
        ret = pd.concat(concat_l, axis=1).iloc[max_lookback+1:, :]\
            .reset_index(drop=True)

        if self.create_rolling_diffs:
            diff_d = {}
            for i in range(1, len(lookback)):
                lb = lookback[i]
                lb_prev = lookback[i-1]
                diff_d.update(
                    {f'{c.split("_ew_rolling")[0]}_ew_rolling_mean_diff_{lb}'
                     f'_{lb_prev}':
                         lambda x: x[c] - x[f'{c.split("_ew_rolling")[0]}'
                         f'_ew_rolling_mean_{lb_prev}'] for c in
                     ew_mean.columns if c.endswith(str(lb))})
            ret = ret.assign(**diff_d)
        return ret

    def _dig_into_rolling_features(self, _df):
        return {str(int(self.get_p_id(_df))):
            self.__dig_into_rolling_features(_df.loc[:, self.x_cols])}

    def _get_noise_augmented_features(self, n_enrich=None):
        if n_enrich is None:
            # calculate required n_enrich
            batchsize = cfg.keras_cfg['rnn_params']['batch_size']
            n_enrich = batchsize // self.downsample_rate - 1

        noisy_dfs = []
        col_filter = [col for col in self.x_cols if 'profile' not in col]
        for n in range(1, n_enrich+1):
            np.random.seed(n)
            # todo: find appropriate noise std per feature
            dummy_noise_level = \
                np.array([1e-4]*len(col_filter))

            df = self.df.copy()
            tmp = self.df.loc[:, col_filter]
            df.loc[:, col_filter] = \
                tmp * (1 + dummy_noise_level * np.random.randn(*tmp.shape))
            df.loc[:, self.PROFILE_ID_COL] += (100*n)
            noisy_dfs.append(df)
        np.random.seed(cfg.data_cfg['random_seed'])
        return noisy_dfs

    def __get_noise_augmented_features(self):
        pass

    def generate_cv_splitter(self, df, p_ids_as_testsets=[]):
        """DEPRECATED: Since BayesOpt requires at least 2 cv splits,
        this function will
        return a cv object that returns indices where the provided profile IDs
        act as testsets. The given df is expected to hold the original train
        and testset but not the validation set as it is part of the early
        stopping criteria.
        """
        assert len(p_ids_as_testsets) > 1, 'provide at least two p_ids that ' \
                                           'shall act as testsets!'
        df.loc[:, 'cv_split'] = np.NaN
        for p_enum, p_id in enumerate(p_ids_as_testsets):
            df.loc[df[self.PROFILE_ID_COL] == p_id, ['cv_split']] = p_enum
        df.fillna({'cv_split': -1}, inplace=True)

        ps = PredefinedSplit(test_fold=df['cv_split'].values)
        return ps

    def get_featurized_sets(self):
        return self.tra_df, self.val_df, self.tst_df

    def get_batch_size(self):
        batch_size = len(self.tra_df[self.PROFILE_ID_COL].unique().tolist()) * \
                     self.downsample_rate
        return batch_size


class PipedDataManager(DataManager):
    """This DataManager class uses scikit learn pipelines and feature unions
    in order to transform data."""

    def __init__(self, path, create_holdout=True):
        super().__init__(path, create_holdout)

        # column management
        self.cl = ColumnManager(self.df)
        self.cl.x_cols = cfg.data_cfg['Input_param_names']

        # build pipeline building blocks
        simple_trans_y = ('y_transform',
                          SimpleTransformer(np.sqrt, np.square, self.cl.y_cols))
        #diff_y = ('y_transform', DiffPerProfileFeatures(self.cl.y_cols))
        featurize_union = FeatureUnion([simple_trans_y,  # simple_trans_y alternatively
                                        ('identity_x', Router(self.cl.x_cols)),
                                       ('lag_feats_x',
                                        LagFeatures(self.cl.x_cols)),
                                       ('rolling_feats_x',
                                        RollingFeatures(self.cl.x_cols,
                                                        lookback=100)),
                                        ('start_of_profile',
                                         SimpleTransformer(
                                             self.indicate_start_of_profile,
                                             None, [self.PROFILE_ID_COL])),
                                        ('i_q_sqrd+i_d_sqrd',
                                         SimpleTransformer(
                                             self.sum_of_squares,
                                             None, ['i_q', 'i_d'])
                                         ),
                                        ('u_q_sqrd+u_d_sqrd',
                                         SimpleTransformer(
                                             self.sum_of_squares,
                                             None, ['u_q', 'u_d'])
                                         )
                                        ])

        featurize_pipe = FeatureUnionReframer.make_df_retaining(featurize_union)

        col_router_pstart = Router([self.START_OF_PROFILE_COL])
        col_router_y = Router(self.cl.y_cols)
        scaling_union = FeatureUnion([('scaler_x', Scaler(StandardScaler(),
                                                          self.cl,
                                                          select='x')),
                                     ('scaler_y', Scaler(StandardScaler(),
                                                         self.cl, select='y')),

                                     ('start_of_profile', col_router_pstart)
                                     ])
        scaling_pipe = FeatureUnionReframer.make_df_retaining(scaling_union)

        poly_union = make_union(Polynomials(degree=2, colmanager=self.cl),
                                col_router_pstart, col_router_y)
        poly_pipe = FeatureUnionReframer.make_df_retaining(poly_union)

        self.pipe = Pipeline([
            ('feat_engineer', featurize_pipe),
            ('cleaning', DFCleaner()),
            ('scaler', scaling_pipe),
            #('poly', poly_pipe),
            ('ident', None)
        ])

        self.resampler = None  # Post-processing

    @property
    def x_cols(self):
        return self.cl.x_cols

    @property
    def y_cols(self):
        return self.cl.y_cols

    @property
    def tra_df(self):
        # todo: This needs testing
        sub_df = super().tra_df
        self.cl.update(sub_df)
        return sub_df

    @measure_time
    def get_featurized_sets(self):
        print('build dataset..')
        tra_df = self.tra_df
        tst_df = self.tst_df
        val_df = self.val_df

        tra_df = self.pipe.fit_transform(tra_df)
        tst_df = self.pipe.transform(tst_df)
        if val_df is not None:
            val_df = self.pipe.transform(val_df)

        self.cl.update(tra_df)
        return tra_df, val_df, tst_df

    def get_featurized_resampled_sets(self):
        tra, val, tst = self.get_featurized_sets()
        batch_size = cfg.keras_cfg['rnn_params']['batch_size']
        self.resampler = ReSamplerForBatchTraining(batch_size=batch_size)
        tra = self.resampler.fit_transform(tra)  # there is nothing fitted
        val = self.resampler.transform(val)
        tst = self.resampler.transform(tst)
        return tra, val, tst

    def inverse_transform(self, pred):
        # reverse post-processing
        pred = self.resampler.inverse_transform(pred)

        # reverse pipeline
        simple_transformer = {k: v for k, v in
                              self.pipe
                                  .named_steps['feat_engineer']
                                  .named_steps['union']
                                  .transformer_list}['y_transform']

        scaler = {k: v for k, v in
                  self.pipe
                      .named_steps['scaler']
                      .named_steps['union']
                      .transformer_list}['scaler_y']

        reduced_pipe = make_pipeline(simple_transformer, scaler)

        inversed = pd.DataFrame(reduced_pipe.inverse_transform(pred),
                                columns=self.cl.y_cols)
        return inversed

    def plot(self):
        import matplotlib.pyplot as plt
        self.df[[c for c in self.x_cols if 'rolling' in c] + self.y_cols]\
            .plot(subplots=True, sharex=True)
        plt.show()

    def indicate_start_of_profile(self, s):
        """Returns a DataFrame where the first observation of each new profile
        id is indicated with True."""
        assert isinstance(s, pd.DataFrame)
        assert s.columns == self.PROFILE_ID_COL
        return pd.DataFrame(data=~s.duplicated(),
                            columns=[self.START_OF_PROFILE_COL])

    @staticmethod
    def sum_of_squares(df):
        """ Return a DataFrame with a single column that is the sum of all
        columns squared"""
        assert isinstance(df, pd.DataFrame)
        colnames = ["{}^2".format(c) for c in list(df.columns)]
        return pd.DataFrame(data=np.square(df).sum(axis=1),
                            columns=["+".join(colnames)])


class SimpleTransformer(BaseEstimator, TransformerMixin):
    """Apply given transformation."""
    def __init__(self, trans_func, untrans_func, columns):
        self.transform_func = trans_func
        self.inverse_transform_func = untrans_func
        self.cols = columns
        self.out_cols = []

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        x = self._get_selection(x)
        ret = self.transform_func(x) if callable(self.transform_func) else x
        self.out_cols = list(ret.columns)
        return ret

    def inverse_transform(self, x):
        return self.inverse_transform_func(x) \
            if callable(self.inverse_transform_func) else x

    def _get_selection(self, df):
        assert isinstance(df, pd.DataFrame)
        assert set(self.cols).issubset(set(df.columns)),\
            '{} is not in {}'.format(self.cols, df.columns)
        return df[self.cols]

    def get_feature_names(self):
        return self.out_cols


class Router(SimpleTransformer):
    """SimpleTransformer with transformation functions being None"""
    def __init__(self, columns):
        super().__init__(None, None, columns=columns)


class DiffPerProfileFeatures():
    """Returns the differences of given data"""
    def __init__(self, cols):
        self.cols = cols
        self.out_cols = []  # should be same as cols here
        self.first_row = None

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        x = self._get_selection(x)
        # todo: This doesnt work. Low Prio
        """ 
        df_per_profile = [x[x[DataManager.PROFILE_ID_COL]==lbl] for lbl in x[
            DataManager.PROFILE_ID_COL].unique().tolist()]
        """

        self.first_row = x.iloc[0, :]
        ret = x.diff(periods=1)
        self.out_cols = list(ret.columns)
        return ret

    def inverse_transform(self, x):
        x = pd.DataFrame(x, columns=self.out_cols)
        x.iloc[0, :] = self.first_row
        return x.cumsum()

    def _get_selection(self, df):
        assert isinstance(df, pd.DataFrame)
        assert set(self.cols).issubset(set(df.columns)),\
            '{} is not in {}'.format(self.cols, df.columns)
        return df[self.cols]

    def get_feature_names(self):
        return self.out_cols


class LagFeatures(BaseEstimator, TransformerMixin):
    """This Transformer adds arithmetic variations between current and lag_x
    observation"""
    def __init__(self, columns, lookback=1):
        self.lookback = lookback
        self.cols = columns
        self.transformed_cols = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = self._get_selection(X)
        dfs = []
        for lback in range(1, self.lookback + 1):
            lag_feats = {'lag{}': X.shift(lback),
                         'lag{}_diff': X.diff(periods=lback),
                         }
            lag_feats['lag{}_abs'] = abs(lag_feats['lag{}_diff'])
            lag_feats['lag{}_sum'] = X + lag_feats['lag{}']

            lag_feats = {key.format(lback): value for key, value
                         in lag_feats.items()}
            # update columns
            for k in lag_feats:
                lag_feats[k].columns = ['{}_{}'.format(c, k) for c in
                                        X.columns]

            dfs.append(pd.concat(list(lag_feats.values()), axis=1))
        df = pd.concat(dfs, axis=1)
        self.transformed_cols = list(df.columns)
        return df

    def _get_selection(self, df):
        assert isinstance(df, pd.DataFrame)
        return df[self.cols]

    def get_feature_names(self):
        return self.transformed_cols


class RollingFeatures(BaseEstimator, TransformerMixin):
    """This Transformer adds rolling statistics"""
    def __init__(self, columns, lookback=10):
        self.lookback = lookback
        self.cols = columns
        self.transformed_cols = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = self._get_selection(X)
        feat_d = {'std': X.rolling(self.lookback).std(),
                  'mean': X.rolling(self.lookback).mean(),
                  # 'sum': X.rolling(self.lookback).sum()
                  }
        for k in feat_d:
            feat_d[k].columns = \
                ['{}_rolling{}_{}'.format(c, self.lookback, k) for
                 c in X.columns]
        df = pd.concat(list(feat_d.values()), axis=1)
        self.transformed_cols = list(df.columns)
        return df

    def _get_selection(self, df):
        assert isinstance(df, pd.DataFrame)
        return df[self.cols]

    def get_feature_names(self):
        return self.transformed_cols


class ReSamplerForBatchTraining(BaseEstimator, TransformerMixin):
    """This transformer sorts the samples according to a
    batch size for batch training"""
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.indices, self.columns = [], []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        # cut the tail
        trunc_idx = len(X) % self.batch_size
        if trunc_idx > 0:
            X = X.iloc[:-trunc_idx, :]

        # reorder
        new_idcs = np.tile(np.arange(self.batch_size), len(X) //
                           self.batch_size)
        assert len(X) == new_idcs.shape[0], \
            "{} != {}".format(len(X), new_idcs.shape[0])
        X.loc[:, 'new_idx'] = new_idcs
        X.sort_values(by='new_idx', ascending=True, inplace=True)
        self.indices = X.index
        X.reset_index(drop=True, inplace=True)
        X.drop(['new_idx'], axis=1, inplace=True)
        self.columns = X.columns
        return X

    def inverse_transform(self, X):
        # columns undefined
        return pd.DataFrame(X, index=self.indices).sort_index()


class Polynomials(BaseEstimator, TransformerMixin):

    def __init__(self, degree, colmanager):
        self.poly = PolynomialFeatures(degree=degree)
        self.out_cols = []
        self.cl = colmanager

    def fit(self, X, y=None):
        X = self._get_selection(X)
        assert isinstance(X, pd.DataFrame)
        self.poly.fit(X, y)
        self.out_cols = self.poly.get_feature_names(input_features=X.columns)
        return self

    def transform(self, X):
        """This transform shall only take Input that has the same columns as
        those this transformer had during fit"""
        X = self._get_selection(X, update=False)
        assert isinstance(X, pd.DataFrame)
        X = self.poly.transform(X)
        ret = pd.DataFrame(X, columns=self.out_cols)
        return ret

    def _get_selection(self, df, update=True):
        assert isinstance(df, pd.DataFrame)
        if update:
            self.cl.update(df)
        return df[self.cl.x_cols]

    def get_feature_names(self):
        return self.out_cols


class DFCleaner(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        X.dropna(inplace=True)
        X.reset_index(drop=True, inplace=True)
        return X


class IdentityEstimator(BaseEstimator, TransformerMixin):
    """This class is for replacing a basic identity estimator with one that
    returns the full input pandas DataFrame instead of a numpy arr
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


class Scaler(BaseEstimator, TransformerMixin):
    """scales selected columns only with given scaler.
    Parameter 'select' is either 'x' or 'y' """
    def __init__(self, scaler, column_manager, select='x'):
        self.cl = column_manager
        self.scaler = scaler
        self.select = select
        self.cols = []

    def fit(self, X, y=None):
        X = self.get_selection(X)
        self.scaler.fit(X, y)
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        X = X[self.cols]
        return self.scaler.transform(X)

    def inverse_transform(self, X):
        return self.scaler.inverse_transform(X)

    def get_selection(self, df):
        assert isinstance(df, pd.DataFrame)
        if self.select.lower() == 'x':
            self.cl.update(df)
            self.cols = self.cl.x_cols
        elif self.select.lower() == 'y':
            self.cols = self.cl.y_cols
        else:
            raise NotImplementedError()
        return df[self.cols]

    def get_feature_names(self):
        return self.cols


class FeatureUnionReframer(BaseEstimator, TransformerMixin):
    """Transforms preceding FeatureUnion's output back into Dataframe"""
    def __init__(self, feat_union, cutoff_transformer_name=True):
        self.union = feat_union
        self.cutoff_transformer_name = cutoff_transformer_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, np.ndarray)
        if self.cutoff_transformer_name:
            # todo Warum haben wir sechs Spalten mehr?
            cols = [c.split('__')[1] for c in self.union.get_feature_names()]
        else:
            cols = self.union.get_feature_names()
        df = pd.DataFrame(data=X, columns=cols)
        return df

    @classmethod
    def make_df_retaining(cls, feature_union):
        """With this method a feature union will be returned as a pipeline
        where the first step is the union and the second is a transformer that
        re-applies the columns to the union's output"""
        return Pipeline([('union', feature_union),
                         ('reframe', cls(feature_union))])


