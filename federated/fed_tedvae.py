#7.16. In this version, it is included the code to load ACIC dataset.

import argparse


import sys
import os

sys.path.insert(0, os.getcwd())
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorflow_addons as tfa

from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
import datetime
import time
import matplotlib.pyplot as plt
from tabulate import tabulate
import pickle
import gc
from typing import List, Tuple, Union, Callable

if 'federated_tedvae/federated' in os.getcwd():
    os.chdir('..')

from datasets.dataset import MultiNodeDataset
from federated.models import FedTEDVAE
from common_utils.callbacks import *
from common_utils.plots import plot_history

physical_devices = tf.config.list_physical_devices('GPU')
print('physical_devices', physical_devices)
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # print('Invalid device or cannot modify virtual devices once initialized')
    pass

# tf.debugging.set_log_device_placement(True)

import pandas as pd

import warnings

warnings.filterwarnings('ignore')


def train_structure(
        datasets_train: List[np.ndarray],
        dataset_validation: np.ndarray,
        dataset_test: np.ndarray,
        y_scaler: StandardScaler,
        columns: List[str],
        i_exp: int,
        hparams: int,
        federated_strategy: str,
        latent_dim_t: int,
        latent_dim_c: int,
        latent_dim_y: int,
        data_types: List[str],
        optimizer: tf.optimizers.Optimizer,
        num_layers: int,
        num_neurons: int,
        num_epochs: int,
        num_epochs_per_fed_average: int,
        num_domains: int,
        loss_weights: Union[dict, pd.DataFrame],
        print_freq: int,
        print_0: int,
        print_last: int,
        results_path: Path,
        verbose: int,
        run_eagerly: bool):
    _tic1 = time.perf_counter()

    tf.keras.backend.clear_session()
    FedTV = FedTEDVAE(
        federated_strategy=federated_strategy,
        num_domains=num_domains,
        latent_dim_t=latent_dim_t,
        latent_dim_c=latent_dim_c,
        latent_dim_y=latent_dim_y,
        data_types=data_types,
        num_layers=num_layers,
        num_neurons=num_neurons,
        num_epochs_per_fed_average=num_epochs_per_fed_average,
        loss_weights=loss_weights
    )

    # optimizer = tf.optimizers.Adam(learning_rate=lr_schedule, clipnorm=10.)
    FedTV.compile(optimizer=optimizer, run_eagerly=run_eagerly)  # true python false C

    # ds_train = tf.data.Dataset.from_tensor_slices(dataset_train).batch(2056*4).prefetch(tf.data.AUTOTUNE)
    # ds_train = tf.data.Dataset.from_tensor_slices(dataset_train).prefetch(tf.data.AUTOTUNE)
    ds_train = []
    for ds in datasets_train:
        ds = np.hstack((ds[:, :-4], ds[:, -1][..., np.newaxis]))
        ds_train.append(ds)

    ds = np.hstack((dataset_validation[:, :-4], dataset_validation[:, -1][..., np.newaxis]))
    ds_val = [ds]*num_domains

    ds_train_val = ds_train + ds_val

    verbose_tf = 0
    if verbose > 0:
        print('training '+ federated_strategy + '...')
        verbose_tf = verbose - 1

    with tf.device('GPU'):
        model_history = FedTV.fit(
            ds_train_val, epochs=num_epochs, verbose=verbose_tf, batch_size=1000,
            callbacks=[
                tf.keras.callbacks.TerminateOnNaN(),
                PrintOutputCallback(print_freq=print_freq, print_0=print_0, epochs=num_epochs, print_last=print_last),
                FederatedAverageCallback()
            ]
        )

    _tac1 = time.perf_counter() #timer

    if verbose > 1:
        print('training time (', federated_strategy, '): ', str(_tac1 - _tic1))


    model_history = pd.DataFrame(model_history.history)

    # TRAIN
    features_train = []
    y_factual_train = []
    t_factual_train = []
    ite_train_list = []
    for ds in datasets_train:
        features_train.append(ds[ds[:, -1].astype('bool'), :-6])
        y_factual_train.append(np.reshape(ds[ds[:, -1].astype('bool'), columns.index('y')], (-1, 1)))
        t_factual_train.append(np.reshape(ds[ds[:, -1].astype('bool'), columns.index('t')], (-1, 1)))
        ite_train_list.append(ds[ds[:, -1].astype('bool'), columns.index('ite')])

    # TEST
    dataset_test = dataset_test[dataset_test[:, -1]==1]
    features_test = [dataset_test[:, :-6]] *2
    ite_test_list = [dataset_test[:, columns.index('ite')]]*2
    _tac2 = time.perf_counter() #timer
    if verbose > 1:
        print('data processing time: ', str(_tac2 - _tac1))

    with tf.device('GPU'):
        pehe_train = FedTV.validate_train(features_train, ite_train_list, y_scaler, y_factual_train, t_factual_train)
        pehe_test = FedTV.validate_test(features_test, ite_test_list, y_scaler)

    tf.keras.backend.clear_session()
    gc.collect()
    del FedTV

    _tac3 = time.perf_counter()
    if verbose > 1:
        print('validation time(', federated_strategy, '): ', str(_tac3 - _tac2))

    file = open(results_path / 'dict_dset_{}_{}_params_{}'.format(i_exp, federated_strategy, hparams), 'wb')


    res = {'pehe_train': pehe_train, 'pehe_test': pehe_test, 'model_history': model_history}


    pickle.dump(res, file)
    file.close()


def load_datasets(i_exp: int, data_path: str, data_setting: str, presaved_path: Path, train_test_val_split: List[int], treatment_split: str):
    data_path = Path(data_path)
    data = MultiNodeDataset(i_exp, data_setting, data_path, train_test_val_split, treatment_split)

    if not os.path.exists(presaved_path):
        os.makedirs(presaved_path)

    file = open(presaved_path / 'data_{}'.format(i_exp), 'wb')
    pickle.dump(data, file)
    file.close()

    return data



def run_experiment(i_exp, n_nodes, hyperparams, federated_strategies, data_path, data_setting, presaved_datapath, results_path,
train_test_val_split, treatment_split,verbose=1):
    results = dict()
    n_epochs= hyperparams.loc[0, 'n_epochs']
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(.1e-2,
                                                                   decay_steps=n_epochs,
                                                                   decay_rate=0.99)

    optimizer = tfa.optimizers.AdamW(weight_decay=1e-4, learning_rate=lr_schedule, clipnorm=10.)

    tic = time.perf_counter()
    if verbose > 2:
        print('----')
        print('----')
        print('----')
        print('Dataset: {}'.format(i_exp))

    #PRESAVED DATA LOCATION DEPENDS ON DATA SETTING
    presaved_path = Path(presaved_datapath) / data_setting / treatment_split


    if os.path.exists(presaved_path / 'data_{}'.format(i_exp)):
        print('DATASET PRESAVED')
        file = open(presaved_path / 'data_{}'.format(i_exp), 'rb')
        data = pickle.load(file)
        file.close()
    else:
        data= load_datasets(i_exp, data_path, data_setting, presaved_path, train_test_val_split, treatment_split)

    data.scale_y_distributed()
    data.pad_datasets()

    for i_param, params in hyperparams.iterrows():
        for federated_strategy in federated_strategies:

            # tf.keras.backend.clear_session()
            train_structure(datasets_train=data.datasets_train,
                            dataset_validation=data.dataset_validation,
                            dataset_test=data.dataset_test,
                            y_scaler=data.y_scaler, columns=data.columns,
                            i_exp=i_exp,
                            hparams=i_param,
                            federated_strategy=federated_strategy,
                            latent_dim_t=LD_t, latent_dim_c=LD_c, latent_dim_y=LD_y,
                            data_types=data.types,
                            optimizer=optimizer,
                            num_layers=NL, num_neurons=NN,
                            num_epochs=params['n_epochs'].astype('int'),
                            num_epochs_per_fed_average=params['n_epochs_fed_avg'],
                            num_domains=n_nodes,
                            loss_weights=params,
                            print_freq=PR_freq, print_0=print_0, print_last=print_last,
                            results_path = results_path,
                            verbose=verbose,
                            run_eagerly=EAGER)

            file = open(results_path / 'dict_dset_{}_{}_params_{}'.format(i_exp, federated_strategy, i_param), 'rb')
            res_dict = pickle.load(file)
            file.close()

            treated_train = [sum(ds[:, data.columns.index('t')]) for ds in data.datasets_train]
            treated_validation = sum(data.dataset_validation[:, data.columns.index('t')])
            treated_test = sum(data.dataset_test[:, data.columns.index('t')])
            if verbose > 0:
                print("TRAIN SPLIT {}".format(treatment_split))
                print("treated_train: {}, treated_validation: {}, treated_test: {}".format(treated_train,
                                                                                           treated_validation,
                                                                                           treated_test))

            if verbose > 0:
                print('PEHE TRAIN {} dset {}'.format( federated_strategy.upper(), i_exp))
                print('domain 1 / domain 2')
                print(res_dict['pehe_train'])

            if verbose > 0:
                print('PEHE TEST {} dset {}'.format(federated_strategy.upper(), i_exp))
                print('domain 1 / domain 2')
                print(res_dict['pehe_test'])

            if PLOT_HISTORY:
                model_history = res_dict['model_history']
                prediction_columns = ['loss_p_t', 'loss_p_y', 'val_loss_p_t', 'val_loss_p_y']
                prediction_columns = ['Domain[{}] {}'.format(i, loss) for loss in prediction_columns for i in range(n_nodes)]

                prediction_history = model_history.loc[:, prediction_columns]
                prediction_history.plot()
                plt.title('params_{} - {} dset {}'.format(i_param,  federated_strategy.upper(), i_exp))
                plt.savefig('./losses_log_{}'.format( federated_strategy))
                plt.show()
                plt.close()

            tac = time.perf_counter()

    if verbose > 0:
        print('Dataset: {}'.format(i_exp + 1))
        print('time of execution: {}'.format(tac - tic))
        print('----')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", "-exp", default=1, type=int, help='Number of the dataset')
    parser.add_argument("--nnodes", "-n", default=2, type=int, help='Number of nodes')
    parser.add_argument("--verbose", "-ver", default=1, type=int, help='verbosity')
    parser.add_argument("--fedavg", "-fs", default='fedavg_all', type=str, help='string of federated strategies, separated by ,')
    parser.add_argument("--data_setting", "-ds", default='tedvae_ihdp', type=str, help='Data settings')
    parser.add_argument("--datapath", "-dp", default='./datasets', type=str, help='Data path')
    parser.add_argument("--presaved_datapath", "-pdp", default='./datasets/presaved_data', type=str, help='Data source')
    parser.add_argument("--hyperpath", "-hpp", default='federated/hyperparams', type=str, help='Hyperparameters dataframe path')
    parser.add_argument("--resultspath", "-rp", default='default_results_path', type=str, help='Results path')
    parser.add_argument("--selectedhp", "-shp", nargs='+', default=[0], type=int, help='Array of selected hyperparams')
    parser.add_argument("--train_test_val_split", "-tvs", default='547,100,100', type=str, help='Train test validation split')
    parser.add_argument("--treatment_split", "-ts", default='51,222,51,222', type=str, help='Treatment split')
    parser.add_argument("--debugger", "-deb", default='False', type=str, help='Debugging mode')

    args = parser.parse_args()

    if args.resultspath == 'default_results_path':
        NOW = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        path = Path("./results/", NOW)
    else:
        path = Path(args.resultspath)
    print(path)
    if not os.path.exists(path):
        os.makedirs(path)

    LD_t = 10  # latent dimension of t space
    LD_c = 20  # latent dimension of c space
    LD_y = 10  # latent dimension of y space
    VERB = 1

    if args.data_setting == 'tedvae_ihdp_a':
        NL = 2
        NN = 64

    elif 'acic' in args.data_setting:
        NL = 2
        NN =  64

    else:
        NL = 3  # number of layers. It is not a Hyperparameter because it has been optimized in centralized version
        NN = 128  # number of neurons per layer. It is not a Hyperparameter because it has been optimized in centralized version

    print_0 = True
    print_last = True
    PLOT_HISTORY = False

    if args.debugger == 'True':
        EAGER = True
        PR_freq = 1
    else:
        EAGER = False
        PR_freq = 200
    # integers: treated_node1, nontreated_node1, treated_node2, nontreated_node2


    federated_strategies = args.fedavg.split(',')
    train_test_val_split = [int(a) for a in args.train_test_val_split.split(',')]

    df_hyperparams = pd.read_csv(Path(args.hyperpath) / 'hyperparams')
    df_hyperparams = df_hyperparams.astype({'n_epochs': 'int32'})

    hp_index = list(args.selectedhp)[0]
    pehe_train = []
    pehe_test = []

    # for hp_index in range(13,21):
    hyperparams = df_hyperparams.iloc[[hp_index], :]

    #run experiment with the selected hyperparameters, make a loop

    run_experiment(i_exp=args.experiment, n_nodes=args.nnodes, hyperparams=hyperparams,
                   federated_strategies = federated_strategies,
                   data_path=args.datapath, data_setting=args.data_setting,
                   presaved_datapath=args.presaved_datapath,
                   results_path = path,
                   train_test_val_split=train_test_val_split, treatment_split=args.treatment_split,
                   verbose=args.verbose)
