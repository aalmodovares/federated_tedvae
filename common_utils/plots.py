import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import pickle


def calculate_stats(values):
    means = np.mean(values, axis=0)
    stds = np.std(values, axis=0)
    stats_str = f"[{means[0]:.6f}, {means[1]:.6f}] +/- [{stds[0]:.6f}, {stds[1]:.6f}]"
    return means, stds, stats_str


def get_one_dict_from_list(dict_list):
    '''

    :param dict_list: list of dictionaries with the same keys
    :return:
     - new_dict: dictionary with the same keys as each dict in dict_list. The values of the last leaves are stored in lists
     - flattened_dict: new dict flattened
     - unique_key_dict: similar to new_dict but not nested
    '''

    new_dict = dict()
    unique_key_dict = dict()

    #INITIALIZE NEW DICT
    for param_0, param_0_dict in dict_list[0].items():
        new_dict[param_0] = dict()
        for param_1, param_1_dict in param_0_dict.items():
            new_dict[param_0][param_1] = {'pehe_train': [], 'pehe_test': []}
            unique_key_dict[param_0 + '_' + param_1 + '_' + 'pehe_train'] = []
            unique_key_dict[param_0 + '_' + param_1 + '_' + 'pehe_test'] = []

    # Populate NEW DICT and UNIQUE KEY DICT
    for d in dict_list:
        for param_0, param_0_dict in d.items():
            for param_1, param_1_dict in param_0_dict.items():
                pehe_train = d[param_0][param_1]['pehe_train']
                pehe_test = d[param_0][param_1]['pehe_test']

                new_dict[param_0][param_1]['pehe_train'].append(pehe_train)
                new_dict[param_0][param_1]['pehe_test'].append(pehe_test)

                unique_key_dict[param_0 + '_' + param_1 + '_' + 'pehe_train'].append(pehe_train)
                unique_key_dict[param_0 + '_' + param_1 + '_' + 'pehe_test'].append(pehe_test)

    flattened_dict = dict()

    for param_0, param_0_dict in new_dict.items():
        for param_1, param_1_dict in param_0_dict.items():

            pehe_train = new_dict[param_0][param_1]['pehe_train']
            pehe_test = new_dict[param_0][param_1]['pehe_test']

            unique_key_dict[param_0 + '_' + param_1 + '_'  + 'pehe_train'] = new_dict[param_0][param_1]['pehe_train']
            unique_key_dict[param_0 + '_' + param_1 + '_'  + 'pehe_test'] = new_dict[param_0][param_1]['pehe_test']

            pehe_train_mean, pehe_train_std, pehe_train_str = calculate_stats(pehe_train)
            pehe_test_mean, pehe_test_std, pehe_test_str = calculate_stats(pehe_test)

            flattened_dict.setdefault('params', []).append(param_0)
            flattened_dict.setdefault('fed_strategy', []).append(param_1)
            flattened_dict.setdefault('pehe_train', []).append(pehe_train_str)
            flattened_dict.setdefault('pehe_test', []).append(pehe_test_str)


    return new_dict, flattened_dict, unique_key_dict



def get_df_from_unique_key_dict(unique_key_dict):
    # Create two separate lists for the first and second columns
    domain1_train = []
    domain1_test = []

    domain2_train = []
    domain2_test = []

    keys_train = [key for key in unique_key_dict if 'train' in key]
    keys_test = [key for key in unique_key_dict if 'test' in key]
    keys_wo_train = [key.replace('_pehe_train', '') for key in keys_train]

    # Iterate over the dictionary and extract the columns
    for key, values in unique_key_dict.items():
        if 'train' in key:
            domain1_train.append([value[0] for value in values])
            domain2_train.append([value[1] for value in values])

        if 'test' in key:
            domain1_test.append([value[0] for value in values])
            domain2_test.append([value[1] for value in values])

    df_1_train = pd.DataFrame(np.array(domain1_train).T, columns=keys_wo_train)
    df_1_train['domain'] = 1
    df_1_train['train'] = 'train'
    df_1_test = (pd.DataFrame(np.array(domain1_test).T, columns=keys_wo_train))
    df_1_test['domain'] = 1
    df_1_test['train'] = 'test'

    df_2_train = (pd.DataFrame(np.array(domain2_train).T, columns=keys_wo_train))
    df_2_train['domain'] = 2
    df_2_train['train'] = 'train'
    df_2_test = (pd.DataFrame(np.array(domain2_test).T, columns=keys_wo_train))
    df_2_test['domain'] = 2
    df_2_test['train'] = 'test'

    df = pd.concat((df_1_train, df_2_train, df_1_test, df_2_test), axis=0)

    df_pivot = pd.melt(df, id_vars=['train', 'domain'], value_vars=keys_wo_train)
    df_pivot.rename(columns={'value': 'PEHE', 'variable': 'fed_case'}, inplace=True)

    return df_pivot

def plot_history(path, i_exp, selected_hp, imbalance, federated_strategy, n_nodes=2):

    results = dict()
    results[f'params_{selected_hp}'] = dict()

    results[f'params_{selected_hp}'] = dict()
    file = open(path / f'dict_dset_{i_exp}_{federated_strategy}_params_{selected_hp}', 'rb')
    res_dict = pickle.load(file)
    file.close()


    model_history = res_dict['model_history']
    prediction_columns = ['loss_p_t', 'loss_p_y', 'val_loss_p_t', 'val_loss_p_y']
    prediction_columns = [f'Domain[{i}] {loss}' for loss in prediction_columns for i in range(n_nodes)]

    prediction_history = model_history.loc[:, prediction_columns]
    prediction_history.plot(logy=True)
    plt.title(f'{federated_strategy.upper()} dset {i_exp}')
    plt.savefig(path / f'losses_log_dset_{i_exp}_{federated_strategy}_params_{selected_hp}')
    plt.show()
    plt.close()

    other_columns = ['loss_reconstruction', 'loss_q_y', 'loss_q_t', 'loss_kl', 'loss_q_zt', 'loss_q_zc', 'loss_q_zy']
    other_columns = [f'Domain[{i}] {loss}' for loss in other_columns for i in range(n_nodes)]

    other_history = model_history.loc[:, other_columns]
    other_history.plot(logy=True)
    plt.title(f'{federated_strategy.upper()} dset {i_exp}')
    plt.savefig(path / f'losses2_log_dset_{i_exp}_{federated_strategy}_params_{selected_hp}')
    plt.show()
    plt.close()