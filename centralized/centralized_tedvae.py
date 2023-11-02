import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, os.getcwd())
from pathlib import Path

import numpy as np
from typing import List

import datetime
import argparse
import pickle

# print('dir: ', os.getcwd())
if 'centralized' in os.getcwd():
    os.chdir('..')
    print('Changed directory to root from centralized.')


import tensorflow as tf

log_dir = "logs/fit/"


from datasets.dataset import MultiNodeDataset
from centralized.local_model import TEDVAE
from common_utils.supervised_metrics import MSE, PEHE, RPEHE
from common_utils.callbacks import *


physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass
import pandas as pd

import warnings
warnings.filterwarnings('ignore')



def load_datasets(i_exp: int, data_path: str, data_setting: str, presaved_path:Path, train_test_val_split: List[int], treatment_split: str):
    data_path = Path(data_path)
    data= MultiNodeDataset(i_exp = i_exp, data_set = data_setting,  data_path = data_path, train_test_val_split=train_test_val_split,
                           treatment_split=treatment_split)


    if not os.path.exists(presaved_path):
        os.makedirs(presaved_path)

    file = open(presaved_path / 'data_{}'.format(i_exp), 'wb')
    pickle.dump(data, file)
    file.close()

    return data

def run_experiment(i_exp, data_path,  data_setting, presaved_datapath, results_path, train_test_val_split, treatment_split,  verbose=1):

    presaved_path = Path(presaved_datapath) / data_setting / treatment_split

    if os.path.exists(presaved_path / 'data_{}'.format(i_exp)):
        print('DATASET PRESAVED')
        file = open(presaved_path / 'data_{}'.format(i_exp), 'rb')
        data = pickle.load(file)
        file.close()

        # data.inverse_scaling_y()
        # data.prune_datasets()

    else:
        data = load_datasets(i_exp, data_path, data_setting, presaved_path, train_test_val_split, treatment_split)

    data.scale_y_distributed(distributed=False)
    data.pad_datasets()
    dataset_train = np.vstack((data.datasets_train))
    dataset_val = data.dataset_validation
    dataset_test = data.dataset_test[data.dataset_test[:, -1] == 1]

    data_types = data.types

    with tf.device("GPU"):
        TV = TEDVAE(
            latent_dim_t=LD_t,
            latent_dim_c=LD_c,
            latent_dim_y=LD_y,
            data_types=data_types,
            num_layers = NL,
            num_neurons = NN
        )
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(1e-3,
                                                                    decay_steps=NE,
                                                                    decay_rate=0.99)
        optimizer = tf.optimizers.AdamW(weight_decay = 1e-4,learning_rate=lr_schedule, clipnorm=10.)

        TV.compile(optimizer=optimizer, run_eagerly=EAGER) #true python false C


        ds_train = np.hstack((dataset_train[:, :-4], dataset_train[:, -1][..., np.newaxis]))
        ds_val = np.hstack((dataset_val[:, :-4], dataset_val[:, -1][..., np.newaxis]))


        ds_train_val = [ds_train] + [np.vstack(([ds_val]*2))]

        if verbose>0:

            model_history = TV.fit(
                ds_train_val, epochs=NE, verbose=verbose, batch_size=1000,
                callbacks=[
                    tf.keras.callbacks.TerminateOnNaN(),
                    PrintOutputCallback(print_freq=50, print_0=print_0, epochs=NE, print_last=print_last),
                ]
            )
        else:
            model_history = TV.fit(
                ds_train_val, epochs=NE, verbose=verbose, batch_size=1000,
                callbacks=[
                    tf.keras.callbacks.TerminateOnNaN(),
                ]
            )

        y_factual_train= np.reshape(ds_train[ds_train[:, -1].astype('bool'), data.columns.index('y')], (-1, 1))
        t_factual_train = np.reshape(ds_train[ds_train[:, -1].astype('bool'), data.columns.index('t')], (-1, 1))

        #TRAIN
        ite_pred, y0_model, y1_model = TV.ite(ds_train[ds_train[:, -1].astype('bool'),:-3], data.y_scaler, y_factual=y_factual_train,
                                              t_factual=t_factual_train)

        pehe_train= RPEHE()(dataset_train[dataset_train[:, -1].astype('bool'), data.columns.index('ite')], ite_pred)


        #TEST
        ite_pred_test, _, _ = TV.ite(dataset_test[:,:-6], data.y_scaler)

        pehe_test = RPEHE()(dataset_test[:, data.columns.index('ite')], ite_pred_test)

        model_history = pd.DataFrame.from_dict(model_history.history)

        if verbose>0:
            print(f'PEHE TRAIN: {pehe_train}')
            print(f'PEHE TEST: {pehe_test}')

        file = open(results_path / f'dict_dset_{i_exp}_centralized', 'wb')
        if PLOT_HISTORY:
            res = {'pehe_train': pehe_train, 'pehe_test': pehe_test, 'model_history': model_history}
        else:
            res = {'pehe_train': pehe_train, 'pehe_test': pehe_test}

        pickle.dump(res, file)
        file.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", "-exp", default=1, type=int, help='Number of the dataset')
    parser.add_argument("--verbose", "-ver", default=0, type=int, help='verbosity')
    parser.add_argument("--datapath", "-dp", default='datasets', type=str, help='Data path')
    parser.add_argument("--data_setting", "-ds", default='tedvae_ihdp', type=str, help='Data settings')
    parser.add_argument("--presaved_datapath", "-pdp", default='./datasets/presaved_data', type=str, help='Data source')
    parser.add_argument("--resultspath", "-rp", default='default_results_path', type=str, help='Results path')
    parser.add_argument("--train_test_val_split", "-tvs", default='547,100,100', type=str, help='Train test validation split')
    parser.add_argument("--treatment_split", "-ts", default='51,222,51,222', type=str, help='Treatment split')
    parser.add_argument("--debugger", "-deb", default='False', type=str, help='Debugging mode')



    args = parser.parse_args()

    if args.resultspath == 'default_results_path':
        NOW = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        path = Path(f"./results/{NOW}")
    else:
        path = Path(args.resultspath)
    print(path)
    if not os.path.exists(path):
        os.mkdir(path)

    LD_t = 10  # latent dimension of t space
    LD_c = 20  # latent dimension of c space
    LD_y = 10  # latent dimension of y space
    VERB=1

    if args.data_setting == 'tedvae_ihdp_a': #simpler arquitecture for ihdp a
        NL = 2
        NN = 64

    elif 'acic' in args.data_setting: # simpler also for acic
        NL = 2
        NN =  64

    else:
        NL = 3  # number of layers. It is not a Hyperparameter because it has been optimized in centralized version
        NN = 128  # number of neurons per layer. It is not a Hyperparameter because it has been optimized in centralized version

    train_test_val_split = [int(a) for a in args.train_test_val_split.split(',')]

    if args.debugger=='True':
        NE = 2
        PR_freq = 1
        EAGER = True
    else:
        NE = 200
        PR_freq = 50
        EAGER = False

    print_0 = True
    print_last = True
    PLOT_HISTORY = False

    run_experiment(i_exp=args.experiment,  data_path=args.datapath, data_setting=args.data_setting, presaved_datapath=args.presaved_datapath,
                   results_path=path,
                   train_test_val_split=train_test_val_split, treatment_split=args.treatment_split,
                   verbose=args.verbose)

