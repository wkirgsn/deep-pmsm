import sqlite3
import argparse
from os.path import join
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import model_from_json

import preprocessing.config as cfg
import preprocessing.file_utils as futils
from preprocessing.data import LightDataManager


def paper_plot():
    cnn_stator_id = 'ebb3'
    cnn_rotor_id = '2950'
    rnn_stator_id = 'f334'
    rnn_rotor_id = 'c329'

    plt.figure(figsize=(2*5, 2*2))
    hp_reports = futils.HyperparameterSearchReport()
    hp_reports.read_search(cnn_stator_id)
    hp_reports.read_search(cnn_rotor_id)
    hp_reports.read_search(rnn_stator_id)
    hp_reports.read_search(rnn_rotor_id)

    # hack: cut off first 50 iterations in RNN Stator search
    tab = hp_reports.hp_searches[rnn_stator_id]
    hp_reports.hp_searches[rnn_stator_id] = \
        tab[tab.n_iter >= 50].reset_index(drop=True)

    plt.subplot(2, 2, 1)
    hp_reports.plot_convergence(rnn_stator_id, 'Stator (RNN)')
    plt.ylabel(r'MSE in $\mathrm{K^2}$ (RNN)')
    plt.xlabel('')
    plt.title('')
    plt.subplot(2, 2, 2)
    hp_reports.plot_convergence(rnn_rotor_id, 'Rotor (RNN)')
    plt.xlabel('')
    plt.ylabel('')
    plt.legend().remove()
    plt.title('')

    plt.subplot(2, 2, 3)
    hp_reports.plot_convergence(cnn_stator_id,  'Stator (CNN)')
    plt.legend().remove()
    plt.title('')
    plt.xlabel('Stator search iteration')
    plt.ylabel(r'MSE in $\mathrm{K^2}$ (CNN)')
    plt.subplot(2, 2, 4)
    hp_reports.plot_convergence(cnn_rotor_id,  'Rotor (CNN)')
    plt.xlabel('Rotor search iteration')
    plt.ylabel('')
    plt.legend().remove()
    plt.title('')

    # find best performing models
    hp_reports.plot_best_models_performance(rnn_rotor_id, rnn_stator_id)
    hp_reports.plot_best_models_performance(cnn_rotor_id, cnn_stator_id)

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize performance of the '
                                                 'given model uid.')
    parser.add_argument('-s', '--stator_id', required=False,
                        help='The 4-digit id in hex of the experiment on '
                             'stator temperatures')
    parser.add_argument('-r', '--rotor_id', required=False,
                        help='The 4-digit id in hex of the experiment on '
                             'rotor temperatures')
    parser.add_argument('-p', '--paper_plot', required=False,
                        action='store_true',
                        help='Flag for ignoring given IDs and instead plot '
                             'four predefined searches for the IEMDC Paper')
    args = parser.parse_args()

    sns.set_context('paper')
    sns.set_style('whitegrid')

    if args.paper_plot:
        paper_plot()
    else:
        assert args.rotor_id is not None and args.stator_id is not None
        hp_reports = futils.HyperparameterSearchReport()
        hp_reports.read_search(args.rotor_id)
        hp_reports.read_search(args.stator_id)

        try:
            print('Plot stator temperature convergence ..')
            plt.figure(figsize=(5, 2.4))
            hp_reports.plot_convergence(args.stator_id)
            print('Plot rotor temperature convergence ..')
            plt.figure(figsize=(5, 2.4))
            hp_reports.plot_convergence(args.rotor_id)
            plt.show()
        except Exception:
            print('plot failed')
