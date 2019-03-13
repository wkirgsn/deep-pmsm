"""Author: WKirgsn, 2018
Hyper Parameter Tuning by Bayesian Optimization through seed trials instead
of Cross-Validation"""
from collections import defaultdict
import gc
import os
import uuid
import sqlite3
import copy
import inspect
from datetime import datetime
import pandas as pd
from skopt import Optimizer
from skopt.callbacks import check_callback
from skopt.space import check_dimension
from skopt.utils import point_asdict, eval_callbacks
from sklearn.utils import check_random_state
from sklearn.externals.joblib import Parallel, delayed
from skopt.callbacks import CheckpointSaver
from skopt import load as skopt_load

from preprocessing import config as cfg
from preprocessing.file_utils import TrialReports


class DataBaseDumperCheck:
    """Simply checks if it was acum_iters times called"""
    def __init__(self):
        self.acum_iters = 1
        self.iter_cnt = 0

    def check(self):
        self.iter_cnt += 1
        return self.iter_cnt % self.acum_iters == 0


class BayesSearchTrials:
    """Inspired by Scikit-Optimize's BayesSearchCV"""

    SEED = cfg.data_cfg['random_seed']
    TABLE_SCHEMES = {'bayes_opt_results': ['bayes_search_id text',
                                           'n_iter integer',
                                           'model_uids text',
                                           'mean_score real',
                                           'best_score real',
                                           'start_time text',
                                           'end_time text']}
    gc.enable()

    def __init__(self, model_func, search_spaces, optimizer_kwargs=None,
                 n_iter=50, fit_params=None,
                 init_params=None,
                 predict_params=None,
                 score_params=None, inverse_params=None, data_manager=None,
                 n_jobs=1, n_seeds=10,
                 n_points=1, verbose=0, random_state=None,
                 return_train_score=False,
                 checkpoint=None, uid='aaaa',
                 known_evals=None):

        self.search_spaces = search_spaces
        self.n_iter = n_iter
        self.n_points = n_points
        self.random_state = random_state
        self.optimizer_kwargs = optimizer_kwargs
        self._check_search_space(self.search_spaces)
        self.n_seeds = n_seeds

        self.model_func = model_func
        self.init_params = init_params
        self.fit_params = fit_params
        self.predict_params = predict_params
        self.score_params = score_params
        self.inverse_params = inverse_params
        self.dm = data_manager

        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state
        self.return_train_score = return_train_score

        self.opt_results = {'mean_test_score': [],
                            'trial_reports': []}
        self.checkpoint = checkpoint
        self.known_evals = known_evals
        self.uid = uid

        self.specs = {'args': {}}
        if cfg.debug_cfg['DEBUG']:
            self.n_iter = 1
            self.n_seeds = 1
        """{"args": copy.copy(inspect.currentframe().f_locals),
             "function": inspect.currentframe().f_code.co_name}
        del self.specs['args']['self']
        del self.specs['args']['model_func']  # unpicklable"""

    @staticmethod
    def load_checkpoint(uid):
        checkpoint = None
        if uid is not None:
            optimizer_uid = uid
            # check for existing checkpoint
            try:
                checkpoint = skopt_load(
                    os.path.join(cfg.data_cfg['model_dump_path'],
                                 'opt_' + optimizer_uid + '.pkl'))
                print(f'Loaded checkpoint for Optimizer uid : {optimizer_uid}')
            except FileNotFoundError:
                print('Checkpoint for specified uid not found.')

        if checkpoint is None:
            optimizer_uid = str(uuid.uuid4())[:4]
            print('Create new optimizer.')
            print('Optimizer UID: {}'.format(optimizer_uid))

        optimizer_callbacks = \
            [CheckpointSaver(os.path.join(cfg.data_cfg['model_dump_path'],
                                          'opt_' + optimizer_uid + '.pkl'),
                             store_objective=False)]
        return checkpoint, optimizer_uid, optimizer_callbacks

    @staticmethod
    def _check_search_space(search_space):
        """Checks whether the search space argument is correct"""

        if len(search_space) == 0:
            raise ValueError(
                "The search_spaces parameter should contain at least one"
                "non-empty search space, got %s" % search_space
            )

        # check if space is a single dict, convert to list if so
        if isinstance(search_space, dict):
            search_space = [search_space]

        # check if the structure of the space is proper
        if isinstance(search_space, list):
            # convert to just a list of dicts
            dicts_only = []

            # 1. check the case when a tuple of space, n_iter is provided
            for elem in search_space:
                if isinstance(elem, tuple):
                    if len(elem) != 2:
                        raise ValueError(
                            "All tuples in list of search spaces should have"
                            "length 2, and contain (dict, int), got %s" % elem
                        )
                    subspace, n_iter = elem

                    if (not isinstance(n_iter, int)) or n_iter < 0:
                        raise ValueError(
                            "Number of iterations in search space should be"
                            "positive integer, got %s in tuple %s " %
                            (n_iter, elem)
                        )

                    # save subspaces here for further checking
                    dicts_only.append(subspace)
                elif isinstance(elem, dict):
                    dicts_only.append(elem)
                else:
                    raise TypeError(
                        "A search space should be provided as a dict or"
                        "tuple (dict, int), got %s" % elem)

            # 2. check all the dicts for correctness of contents
            for subspace in dicts_only:
                for k, v in subspace.items():
                    check_dimension(v)
        else:
            raise TypeError(
                "Search space should be provided as a dict or list of dict,"
                "got %s" % search_space)

    def _prepare_cfg_on_parameters(self, params):
        """Versatile config setting according to asked parameters"""
        config_list = [cfg.data_cfg, cfg.keras_cfg, cfg.lgbm_cfg,
                      self.fit_params,
                      self.predict_params,
                      self.init_params,
                      self.inverse_params]

        # apply new parameters
        for p_key, p_value in params.items():
            for d in config_list:
                if p_key in d and p_value is not None:
                    d[p_key] = p_value

        downsample_rate = params.get('downsample_rate', None)
        if downsample_rate is not None:
            self.dm.downsample_rate = downsample_rate

        lookback = [params.get(i, None) for i in params if 'rolling_lb_' in i]

        if not any(l is None for l in lookback):
            lookback = list(set(int(l) for l in lookback))
            # rebuild datamanager
            self.dm.rolling_lookback = lookback
            cfg.data_cfg['rolling_lookback'] = lookback
            self.dm.reset()
            self.dm.featurize()

            x_train = self.dm.tra_df[self.dm.x_cols + [self.dm.PROFILE_ID_COL]]
            y_train = self.dm.tra_df[self.dm.y_cols + [self.dm.PROFILE_ID_COL]]
            x_val = self.dm.val_df[self.dm.x_cols + [self.dm.PROFILE_ID_COL]]
            y_val = self.dm.val_df[self.dm.y_cols + [self.dm.PROFILE_ID_COL]]
            x_tst = self.dm.tst_df[self.dm.x_cols + [self.dm.PROFILE_ID_COL]]
            y_tst = self.dm.tst_df[self.dm.y_cols]

            self.fit_params['x'] = x_train
            self.fit_params['y'] = y_train
            self.fit_params['validation_data'] = (x_val, y_val)
            self.predict_params['x'] = x_tst
            self.inverse_params['df'] = y_tst

            # reset cache
            self.fit_params['data_cache'] = {}

        # todo: The following is KERAS specific. Work out general implementation
        # cnn (window_size) or rnn (tbptt_len)?
        if 'tbptt_len' in params:
            # we have a RNN and need to manually set the batch_size
            batch_size = self.dm.get_batch_size()
            for d in config_list + [cfg.keras_cfg['rnn_params'], ]:
                if 'batch_size' in d:
                    d['batch_size'] = batch_size
            self.init_params['x_shape'] = (batch_size,
                                           params['tbptt_len'],
                                           len(self.dm.x_cols))
        elif 'window_size' in params:
            # we have a CNN and need to manually set x_shape
            self.init_params['x_shape'] = (params['window_size'],
                                           len(self.dm.x_cols))

    def _run_on_search_space(self, search_space, optimizer, n_points=1):
        """Generate n_points parameter sets and evaluate them in parallel.
        Parallel computing is done across seeds, not exploration points"""

        # get parameter values to evaluate
        exploration_points = optimizer.ask(n_points=n_points)
        params_dict_l = [point_asdict(search_space, p) for p in
                        exploration_points]

        seed_list = [x + self.SEED for x in range(self.n_seeds)]

        if self.verbose > 0:
            n_candidates = len(params_dict_l)
            s1 = 's' if self.n_seeds > 1 else ''
            s2 = 's' if n_candidates > 1 else ''
            s3 = 's' if n_candidates*self.n_seeds > 1 else ''
            print(f"Fitting {self.n_seeds} seed{s1} for each of {n_candidates} "
                  f"candidate{s2}, "
                  f"totalling {n_candidates*self.n_seeds} fit{s3}")

        trial_reports = []

        for parameters in params_dict_l:
            self._prepare_cfg_on_parameters(parameters)

            gc.collect()
            start_time = datetime.now().strftime("%Y-%m-%d %H:%M")
            reports = Parallel(
                n_jobs=self.n_jobs, verbose=self.verbose,
                pre_dispatch=2*self.n_jobs)(
                delayed(TrialReports.conduct_step)(
                    model_func=self.model_func,
                    seed=seed,
                    init_params=self.init_params,
                    fit_params=self.fit_params,
                    predict_params=self.predict_params,
                    inverse_params=self.inverse_params,
                    dm=self.dm
                ) for seed in seed_list
            )
            trial = TrialReports(reports=reports)
            trial.start_time = start_time
            trial.end_time = datetime.now().strftime("%Y-%m-%d %H:%M")
            trial.print(plot=False)
            trial_reports.append(trial)  # amt trialReports == amt explor. pts.

        mean_scores = [t.get_mean_score() for t in trial_reports]
        self.opt_results['mean_test_score'].extend(mean_scores)
        self.opt_results['trial_reports'].extend(trial_reports)

        # optimizer minimizes objective
        return optimizer.tell(exploration_points, mean_scores)

    def run(self, callback=None):
        """Run TrialReports n_iter times"""
        # check if space is a single dict, convert to list if so
        search_spaces = self.search_spaces
        if isinstance(search_spaces, dict):
            search_spaces = [search_spaces]

        callbacks = check_callback(callback)

        if self.optimizer_kwargs is None:
            self.optimizer_kwargs_ = {}
        else:
            self.optimizer_kwargs_ = dict(self.optimizer_kwargs)
        random_state = check_random_state(self.random_state)
        self.optimizer_kwargs_['random_state'] = random_state

        # Instantiate optimizers for all the search spaces.
        optimizers = []
        for search_space in search_spaces:
            if isinstance(search_space, tuple):
                search_space = search_space[0]
            kwargs = self.optimizer_kwargs_.copy()
            kwargs['dimensions'] = \
                [search_space[k] for k in sorted(search_space.keys())]
            opt = Optimizer(**kwargs)
            if self.checkpoint is not None:
                opt.rng = self.checkpoint.random_state

                # flip sign for falsely optimized evaluation points :_(
                y_iters = [abs(a) for a in list(self.checkpoint.func_vals)]

                print(f'Checkpoint contains {len(y_iters)} iterations.')

                # load checkpoint info into optimizer
                opt.tell(self.checkpoint.x_iters, y_iters)

                self.n_iter -= len(self.checkpoint.x_iters)
            if self.known_evals is not None:
                # fresh optimization, take known evaluations from config into
                #  account
                # todo: The following is rotor<->stator separation specific.
                #  Work out general implementation

                # known_evals is a list of dicts where x & y are tied up
                y_iters_rotor = [space_d.pop('y_rotor', None) for space_d in
                                 self.known_evals]
                y_iters_stator = [space_d.pop('y_stator', None) for space_d in
                                 self.known_evals]

                if any(y is None for y in y_iters_rotor+y_iters_stator):
                    print('Known evaluations were found but there are y '
                          'values missing. Continueing without telling '
                          'optimizer of known evaluations.')
                else:
                    y_iters = y_iters_stator
                    len_iters = len(y_iters)
                    if 'stator' in self.fit_params['y'].columns[0]:
                        print(f'Found {len_iters} known evaluations in config.'
                              f' Telling optimizer the stator outcome.')
                    else:
                        y_iters = y_iters_rotor
                        len_iters = len(y_iters)
                        print(f'Found {len_iters} known evaluations in config.'
                              f' Telling optimizer the rotor outcome.')
                    x_iters = []
                    for space_d in self.known_evals:
                        x_iters.append([space_d[k] for k in
                                        sorted(space_d.keys())])

                    opt.tell(x_iters, y_iters)
                    # self.n_iter -= len(x_iters) # dont reduce n_iters,
                    # as scatter info is missing for these known evaluations

            optimizers.append(opt)
        self.optimizers_ = optimizers  # will save the states of the optimizers

        if self.verbose > 0:
            s1 = 's' if len(search_spaces) > 1 else ''
            s2 = 's' if self.n_iter > 1 else ''
            print(f'Start Bayesian Opt on {len(search_spaces)} '
                  f'search space{s1}, for {self.n_iter} iteration{s2} each.')

        for search_space, optimizer in zip(search_spaces, optimizers):
            n_iter = self.n_iter
            db_dump = DataBaseDumperCheck()
            # do the optimization for particular search space
            while n_iter > 0:
                # when n_iter < n_points, there are fewer points left as
                # usual for evaluation
                n_points_adjusted = min(n_iter, self.n_points)

                optim_result = \
                    self._run_on_search_space(search_space,
                                              optimizer,
                                              n_points=n_points_adjusted)
                n_iter -= self.n_points
                optim_result.specs = self.specs  # overwrite None-Specs

                if eval_callbacks(callbacks, optim_result):
                    break

                # special callback - needs to call this class' function
                if db_dump.check():
                    self.dump_result()

        return self

    def dump_result(self):
        trials = self.opt_results['trial_reports']
        if len(trials) == 0 or cfg.debug_cfg['DEBUG']:
            return

        print(f'\nDumping {len(trials)} optimization result(s)')

        df_d = {'bayes_search_id': self.uid,
                'n_iter': list(range(len(trials))),
                'model_uids': [str(t.get_uids()) for t in trials],
                'mean_score': self.opt_results['mean_test_score'],
                'best_score': [min(t.get_scores()) for t in trials],
                'start_time': [t.start_time for t in trials],
                'end_time': [t.end_time for t in trials],
                }

        table_name = 'bayes_opt_results'
        table_scheme = self.TABLE_SCHEMES[table_name]

        assert set(df_d.keys()) == set([s.split(' ')[0] for s in table_scheme]),\
            'table scheme inconsistence!'

        with sqlite3.connect(cfg.data_cfg['db_path']) as con:
            query = "CREATE TABLE IF NOT EXISTS " + \
                    "{}{}".format(table_name, tuple(table_scheme)) \
                        .replace("'", "")
            con.execute(query)

            # increment n_iter with what can be found in the db for that exp.
            query = "SELECT * FROM {}".format(table_name)
            opt_table = pd.read_sql_query(query, con)
            if self.uid in opt_table.bayes_search_id.unique():
                max_iter = opt_table[opt_table.bayes_search_id ==
                                     self.uid].n_iter.max() + 1
                print(f'Found {max_iter} iter(s) in db for id {self.uid}. '
                      f'Append current results on top.')
                df_d['n_iter'] = list(range(max_iter, max_iter+len(trials)))

            df = pd.DataFrame(df_d)
            entries = [tuple(x) for x in df.values]
            query = f'INSERT INTO {table_name} ' + \
                    'VALUES ({})'.format(', '.join('?' * len(df.columns)))
            con.executemany(query, entries)
        # clear reports
        del df
        del trials
        self.opt_results['trial_reports'] = []
        self.opt_results['mean_test_score'] = []
        gc.collect()
