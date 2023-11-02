"""
IHDP (Infant Health and Development Program) dataset
"""
import copy
# stdlib
import sys
import os
sys.path.insert(0, os.path.dirname(os.getcwd()))
# print(sys.path)
import random
from pathlib import Path
from typing import Any, Tuple, List, Union
from enum import Enum
from sklearn.preprocessing import StandardScaler

# third party
import numpy as np
import pandas as pd
import json

import warnings

# np.random.seed(0)
# random.seed(0)

class DataType(Enum):
    REAL = 1
    BINARY = 2
    CATEGORICAL = 3
    DISCARD = 4


class Dataset:
    def __init__(self, i_exp: int, data_set: str, data_path: Path, preprocess: bool=True) -> None:
        self.data: np.ndarray
        # types are needed to train any factor model
        self._types: List[DataType]
        self._columns: List[str]

        self.data_setting = data_set

        if 'dataset' not in data_path.absolute().as_posix():
            data_path = Path('datasets') / data_path

        self.loader = self.load(i_exp, data_path, preprocess)
        print(data_set + " Dataset {} loaded".format(i_exp))


        # self.train_test_splitter = self.train_test_split(train_test_val_split)

    @property
    def types(self):
        return self._types

    @types.setter
    def types(self, types):
        if self.data is None:
            raise AttributeError("Data has not been loaded yet, use load method")

        self._types = types

    #setter of columns
    @property
    def columns(self):
        return self._columns

    @columns.setter
    def columns(self, columns):
        if self.data is None:
            raise AttributeError("Data has not been loaded yet, use load method")
        self._columns = columns

    def info(self):
        return {
            "types": self.types,
            "size": self.data.shape
        }
    def load_data_npz(self, fname: Path) -> dict:
        """
        Helper function for loading the IHDP data set (adapted from https://github.com/clinicalml/cfrnet)

        Parameters
        ----------
        fname: Path
            Dataset path

        Returns
        -------
        data: dict
            Raw IHDP dict, with X, w, y and yf keys.
        """
        data_in = np.load(fname)
        data = {"X": data_in["x"], "w": data_in["t"], "y": data_in["yf"]}

        data["mu0"] = data_in["mu0"]
        data["mu1"] = data_in["mu1"]

        return data

    def get_one_data_set(self, D: dict, i_exp: int) -> dict:
        """
        Helper for getting the IHDP data for one experiment. Adapted from https://github.com/clinicalml/cfrnet

        Parameters
        ----------
        D: dict or pd.DataFrame
            All the experiment
        i_exp: int
            Experiment number

        Returns
        -------
        data: dict or pd.Dataframe
            dict with the experiment
        """
        D_exp = {}
        D_exp["X"] = D["X"][:, :, i_exp - 1]
        D_exp["w"] = D["w"][:, i_exp - 1 : i_exp]
        D_exp["y"] = D["y"][:, i_exp - 1 : i_exp]

        D_exp["mu0"] = D["mu0"][:, i_exp - 1 : i_exp]
        D_exp["mu1"] = D["mu1"][:, i_exp - 1 : i_exp]

        return D_exp

    def load(self, i_exp: int, data_path: Path, preprocess: bool = True) -> None:

        if self.data_setting=='catenets_ihdp':
             self.load_catenets_data(i_exp, data_path)
        elif self.data_setting=='tedvae_ihdp':
             self.load_tedvae_data(i_exp, data_path, setting='B')
        elif self.data_setting=='tedvae_ihdp_a':
             self.load_tedvae_data(i_exp, data_path, setting='A')
        elif self.data_setting=='acic':
             self.load_acic_data(i_exp, data_path)
        elif self.data_setting=='twins':
             self.load_twins_data(i_exp, data_path)
        elif self.data_setting=='synthetic':
             self.load_synthetic_data(i_exp, data_path)
        else:
            raise NotImplementedError('data not found')

        if preprocess:
            self.remove_categorical()
            self.cast_to_float()
            self.shuffle()
            self.standardize_real()

    def train_test_split(self, train_test_val_split: List[int], n_val_test_treated = 0) -> None:
        '''
        Split the dataset in train, test and validation
        :param train_test_val_split: list with the number of patients in train, test and validation
        :return: None
        self.datasets : list with the dataset of train, test and validation
        '''
        data = self.data

        data_t1 = data[data[:, self.columns.index('t')] == 1]
        data_t0 = data[data[:, self.columns.index('t')] == 0]

        if n_val_test_treated==0:
            # proportion of treated in total:
            p_treated = len(data_t1) / len(data)
            n_val_test_treated = int(train_test_val_split[1] * p_treated)

        n_untreated_val = train_test_val_split[1] - n_val_test_treated
        n_untreated_test = train_test_val_split[2] - n_val_test_treated

        treated_val = data_t1[:n_val_test_treated]
        untreated_val = data_t0[:n_untreated_val]

        treated_test = data_t1[n_val_test_treated:n_val_test_treated*2]
        untreated_test = data_t0[n_untreated_val:n_untreated_val + n_untreated_test]

        treated_train = data_t1[n_val_test_treated*2:]
        untreated_train = data_t0[n_untreated_val + n_untreated_test:]



        #split the data in train, test and validation with the values of train_test_val_split
        train_data = ((np.vstack((treated_train, untreated_train)))[0:train_test_val_split[0]])
        test_data = (np.vstack((treated_test, untreated_test)))
        validation_data = (np.vstack((treated_val, untreated_val)))

        np.random.shuffle(train_data)
        np.random.shuffle(test_data)
        np.random.shuffle(validation_data)


        if sum(train_test_val_split) < len(data):
            warnings.warn('Data used for training, test and validation is smaller than the total of patients')
        if sum(train_test_val_split) > len(data):
            raise ValueError('Data used for training, test and validation is greater than the total of patients')

        self.datasets = [train_data, test_data, validation_data]

    def load_tedvae_data(self, i_exp: int, data_path: Path, setting: str = 'B'):

        dataset_name = 'IHDP_b' if setting=='B' else 'IHDP'

        data_path = data_path / 'TEDVAE_data' / dataset_name

        data = np.loadtxt(data_path / 'ihdp_npci_train_{}.csv'.format(i_exp), delimiter=',', skiprows=1)

        t, y = data[:, 0][:, np.newaxis], data[:, 1][:, np.newaxis]
        mu_0, mu_1, x = data[:, 3][:, np.newaxis], data[:, 4][:, np.newaxis],data[:, 5:]
        pos = np.asarray([mu_0, mu_1]).squeeze().T.astype(np.float32)
        true_ite = mu_1 - mu_0
        x[:, 13] -= 1

        data_test = np.loadtxt(data_path / 'ihdp_npci_test_{}.csv'.format(i_exp), delimiter=',', skiprows=1)
        t_test, y_test = data_test[:, 0][:, np.newaxis], data_test[:, 1][:, np.newaxis]
        mu_0_test, mu_1_test, x_test = data_test[:, 3][:, np.newaxis], data_test[:, 4][:, np.newaxis], data_test[:, 5:]
        pos_test = np.asarray([mu_0_test, mu_1_test]).squeeze().T.astype(np.float32)
        true_ite_test = mu_1_test - mu_0_test
        x_test[:, 13] -= 1

        dataset_train = np.hstack((x, t, y, pos, true_ite))
        dataset_test = np.hstack((x_test, t_test, y_test, pos_test, true_ite_test))


        self.data = np.vstack((dataset_train, dataset_test)).astype(np.float32)
        self.types = ["real" for _ in range(6)] + ["binary" for _ in range(19)]

        self.columns = ['x']*x.shape[1] + ['t', 'y', 'mu0', 'mu1', 'ite']

    def load_catenets_data(self, i_exp:int, data_path: Path):
        """
                Get IHDP train/test datasets with treatments and labels.

                Parameters
                ----------
                data_path: Path
                    Path to the dataset csv. If the data is missing, it will be downloaded.


                Returns
                -------
                X: pd.Dataframe or array
                    The training feature set
                w: pd.DataFrame or array
                    Training treatment assignments.
                y: pd.Dataframe or array
                    The training labels
                training potential outcomes: pd.DataFrame or array.
                    Potential outcomes for the training set.
                X_t: pd.DataFrame or array
                    The testing feature set
                testing potential outcomes: pd.DataFrame of array
                    Potential outcomes for the testing set.
                """
        data_path = data_path / 'CATENets_data' / 'IHDP'

        TRAIN_DATASET = "ihdp_npci_1-100.train.npz"
        TEST_DATASET = "ihdp_npci_1-100.test.npz"

        train_csv = data_path / TRAIN_DATASET
        test_csv = data_path / TEST_DATASET

        data_train = self.load_data_npz(train_csv)
        data_test = self.load_data_npz(test_csv)

        data_train = self.get_one_data_set(data_train, i_exp=i_exp)
        data_test = self.get_one_data_set(data_test, i_exp=i_exp)

        X, y, t, mu0, mu1 = (
            data_train["X"],
            data_train["y"],
            data_train["w"],
            data_train["mu0"],
            data_train["mu1"],
        )
        ite_train = mu1 - mu0

        # y_t and w_t are not expected to be returned because they are not available for new patients
        # they are returned for convenience in case the partition of train/test sets was different
        X_t, y_t, t_t, mu0_t, mu1_t = (
            data_test["X"],
            data_test["y"],
            data_test["w"],
            data_test["mu0"],
            data_test["mu1"],
        )
        ite_test = mu1_t - mu0_t

        dataset_train = np.hstack((X, t, y, mu0, mu1, ite_train))
        dataset_test = np.hstack((X_t, t_t, y_t, mu0_t, mu1_t, ite_test))

        data = np.vstack((dataset_train, dataset_test)).astype(np.float32)
        data[:, 13] -= 1
        self.data = data

        self.types = ["real" for _ in range(6)] + ["binary" for _ in range(19)]
        self.columns = ['x' + str(i+1) for i in range(X_t.shape[1])] + ['t', 'y', 'mu0', 'mu1', 'ite']

    def load_acic_data(self, i_exp: int, data_path: Path, setting: str = '2'):

        data_path = data_path / 'ACIC2016' / 'data_cf_all' / setting

        list_of_files = os.listdir(data_path)

        #Load the data
        # columns "z","y0","y1","mu0","mu1". z is the treatment, y0 and y1 are the potential outcomes with noise, mu0 and mu1 are the potential outcomes without noise
        data_tymu = pd.read_csv(data_path / list_of_files[i_exp-1])

        t = data_tymu['z']

        # define the outcome: y = y1 if t = 1, y = y0 if t = 0
        y = data_tymu['y1'] * t + data_tymu['y0'] * (1 - t)

        mu_0, mu_1 = data_tymu['mu0'], data_tymu['mu1']
        pos = np.asarray([mu_0, mu_1]).squeeze().T.astype(np.float32)
        true_ite = mu_1 - mu_0

        # load the covariates from the parent directory
        data_x = pd.read_csv(data_path / '..' / 'x.csv')

        dataset = np.hstack((data_x, np.array(t).reshape(-1, 1), np.array(y).reshape(-1, 1), pos, np.array(true_ite).reshape(-1, 1)))

        self.data = dataset
        #save types of covariates. Check if the values of the columns are binary (0/1), continuous or categorial (defined by capital letters)
        categorical_positions = [1,20,23]
        self.types = ['binary' if len(np.unique(data_x.iloc[:,i]))==2 else 'categorical' if i in categorical_positions else 'real' for i in range(data_x.shape[1])]
        self.columns = ['x' + str(i+1) for i in range(data_x.shape[1])] + ['t', 'y', 'mu0', 'mu1', 'ite']

    def load_twins_data(self, i_exp: int, data_path: Path):
        """
        Get TWINS train/test datasets with treatments and labels.

        Parameters
        ----------
        data_path: Path
            Path to the dataset csv. If the data is missing, it will be downloaded.
            """
        data_path = data_path / 'TWINS'

        t = pd.read_csv(data_path / 'twin_pairs_T_3years_samesex.csv')
        y = pd.read_csv(data_path / 'twin_pairs_Y_3years_samesex.csv')
        x = pd.read_csv(data_path / 'twin_pairs_X_3years_samesex.csv')

        # change the index of the dataframes to Unnamed: 0 and ignore index.
        t = t.set_index('Unnamed: 0', drop=True)
        y = y.set_index('Unnamed: 0', drop=True)
        x = x.set_index('Unnamed: 0', drop=True)
        #remove Unnamed: 0.1 from x
        x = x.drop(['Unnamed: 0.1', 'infant_id_0', 'infant_id_1'], axis=1)

        # #remove, from x, the columns that have more than 10% of missing values
        # columns_nan = x.columns[x.isna().mean() > 0.1]
        # x = x.drop(columns_nan, axis=1)
        #remove, from x, the rows that have missing values and save the index of the rows that were removed
        rows_nan = x[x.isna().any(axis=1)].index
        x = x.dropna(axis=0)

        #remove, from t and y, the rows removed in x
        t = t.drop(rows_nan, axis=0)
        y = y.drop(rows_nan, axis=0)

        #in this dataframe, we have the potential outcomes mu0 and mu1 in y
        mu0 = y['mort_0']
        mu1 = y['mort_1']

        ite = mu1 - mu0

        #the y column depends of some criteria of selection. At the moment: RCT
        t = np.random.binomial(1, 0.5, size=len(t))
        y = mu1 * t + mu0 * (1 - t)

        #load a dict from covar_desc.txt, where keys and values are in single quotes (')
        with open(data_path / 'covar_type.txt') as json_file:
            covar_type = json.load(json_file)

        types = []
        columns = []

        for column in x.columns:
            col_name = column.split('_')[0] #this line links the bord_0 and _1 with their type. It also removes infant_id
            #check if col_name is part of one of the keys of the dict covar_type
            if col_name in covar_type.keys():
                types.append(covar_type[col_name])
                columns.append(column)

        x = x[columns]


        dataset = np.hstack((x, np.array(t).reshape(-1, 1), np.array(y).reshape(-1, 1), np.array(mu0).reshape(-1, 1), np.array(mu1).reshape(-1,
                                                                                                                                            1),
                             np.array(ite).reshape(-1, 1)))
        self.data = dataset
        self.columns = columns + ['t', 'y', 'mu0', 'mu1', 'ite']
        self.types = types

    def load_synthetic_data(self, i_exp: int, data_path: Path):
        pass
    def cast_to_float(self):
        """
        Cast the data to float32
        :return: None
        """
        self.data = self.data.astype(np.float32)

    def remove_categorical(self):
        data_types = self.types
        data = self.data
        new_data_types = []
        new_data = []
        new_columns = []
        i=0
        for j, type in enumerate(data_types):
            if type !='categorical':
                if new_data == []:
                    new_data = data[:, j][..., np.newaxis]
                else:
                    new_data = np.hstack((new_data, data[:, j][..., np.newaxis]))
                new_data_types.append(type)
                i+=1

                #append column to new columns
                new_columns.append(self.columns[j])

        new_data = np.hstack((new_data, data[:, -5:]))

        self.data = new_data
        self.types = new_data_types
        self.columns = new_columns + ['t', 'y', 'mu0', 'mu1', 'ite']

    def shuffle(self, random_state = None):
        if random_state is not None:
            np.random.seed(random_state)
        np.random.shuffle(self.data)

    def standardize_real(self):
        """
        Standardize the real columns of the dataset
        :return: None
        """
        for i, type in enumerate(self.types):
            if type == 'real':
                mean = np.mean(self.data[:, i])
                std = np.std(self.data[:, i])
                self.data[:, i] = (self.data[:, i] - mean) / std

    def scale_y(self):
        """
        Scale the y column of the dataset
        :return: None
        """
        y = self.data[:, self.columns.index('y')][..., np.newaxis]
        y_scaler = StandardScaler()
        y_scaler.fit(y)

        y_scaled = y_scaler.transform(y)
        self.data[:, self.columns.index('y')] = y_scaled.squeeze()
        self.y_scaler = y_scaler

    def pad_datasets(self):
        '''
        Pad the datasets with zeros to have the same length to include as list in the fit method
        :return: None
        self.datasets has a list of datasets with the same length
        '''
        datasets_copy = copy.deepcopy(self.datasets)
        width = self.datasets[0].shape[1]
        max_len = max([len(self.datasets[i]) for i in range(len(self.datasets))])
        new_columns = copy.deepcopy(self.columns)

        for i, ds in enumerate(self.datasets):
            datasets_copy[i] = np.hstack((datasets_copy[i], np.ones(shape=(len(ds), 1))))
            if len(ds) != max_len:
                if len(ds) > max_len:
                    max_len = len(ds)

        for j, ds in enumerate(datasets_copy):
            difference = max_len - len(ds)
            aux = np.zeros(shape=(difference, width + 1))
            datasets_copy[j] = np.vstack((ds, aux))

        new_columns.append('flag')

        self.datasets = datasets_copy

class MultiNodeDataset(Dataset):

    def __init__(self, i_exp: int, data_set: str, data_path: Path, train_test_val_split: List[int], treatment_split:str,
                 preprocess: bool=True) -> None:
        super().__init__(i_exp, data_set, data_path, preprocess)

        self.data: np.ndarray
        self._types: List[DataType]
        self._columns: List[str]

        self.n_nodes = int(len(treatment_split.split(','))/2)
        self.datasets = List[np.ndarray]
        self.datasets_train = List[np.ndarray]
        self.dataset_validation = np.ndarray
        self.dataset_test = np.ndarray
        self.split_info = List[List[int]]

        self.train_test_split = train_test_val_split
        self.treatment_splitter = self.split_dataset(treatment_split)



        print('Dataset split')


    def split_dataset(self, nodes_split: str):
        """

        :param nodes_split: list of lists: [[untreated_node_1, treated_node_1], [untreated_node_2, treated_node_2], ...]
            nodes_split may also include, as the two last nodes, the nodes that will be used for validation and test

        :self. datasets : list with the dataset of each node
        """
        treated_list = nodes_split.split(',')

        if 'acic' in self.data_setting:
            total_treated = 1392
            total_patients = 4802
            total_untreated = total_patients - total_treated

            treated_proportion =total_treated/total_patients
            treated_validation_and_test = [ np.round(self.train_test_split[1]*treated_proportion),  np.round(self.train_test_split[
                                                                                                                 2]*treated_proportion)]
        elif 'ihdp' in self.data_setting:
            total_treated = 139
            total_patients = 747
            total_untreated = total_patients - total_treated

            treated_validation_and_test = [18,19]
        elif 'twins' in self.data_setting:
            pass
        elif 'jobs' in self.data_setting:
            pass

        else:
            raise ValueError('Invalid data setting')


        untreated_validation = self.train_test_split[1] - treated_validation_and_test[0]
        untreated_test = self.train_test_split[2] - treated_validation_and_test[1]

        nodes_train_split = []
        for i in range(self.n_nodes):
            if treated_list[i*2+1] == 'random':
                node_split = [int(treated_list[i*2])]
            else:
                treated = int(treated_list[i*2])
                untreated = int(treated_list[i*2+1])
                node_split = [untreated, treated]

            nodes_train_split.append(node_split)


        total_split = nodes_train_split + [[untreated_validation, treated_validation_and_test[0]], [untreated_test, treated_validation_and_test[1]]]

        data = self.data # train data, test and validation data is the same for all nodes.

        datasets_list = []
        # for random split:
        for split in total_split:
            if len(split)==1:
                datasets_list.append(data[:split[0]])
                data = data[split[0]:]

        #VALIDATION IS NOT USED FOR TRAINING FOR EARLY STOPPING.
        t_col = data[:, self.columns.index('t')]  # get the binary column `t`
        mask_t1 = t_col == 1  # create a mask for `t`=1
        mask_t0 = t_col == 0  # create a mask for `t`=0


        # split the data based on `t`=1
        data_t1 = data[mask_t1]

        # split the data based on `t`=0
        data_t0 = data[mask_t0]

        start_index_t0 = final_index_t0 = 0  #
        start_index_t1 = final_index_t1 = 0

        for split in total_split:
            if len(split)==1:
                continue

            if split[0] + start_index_t0 >= len(data_t0):
                warnings.warn('Number of untreated in the split greater than total of untreated patients')
                final_index_t0 = len(data_t0)
            else:
                final_index_t0 = start_index_t0 + int(split[0])

            if split[1] + start_index_t1 >= len(data_t1):
                warnings.warn('Number of treated in the split greater than total of treated patients')
                final_index_t1 = len(data_t1)
            else:
                final_index_t1 = start_index_t1 + int(split[1])

            dataset_node_untreated = data_t0[start_index_t0:final_index_t0, :]
            dataset_node_treated = data_t1[start_index_t1:final_index_t1, :]

            start_index_t0 = final_index_t0
            start_index_t1 = final_index_t1

            dataset_node = np.vstack((dataset_node_untreated, dataset_node_treated))
            np.random.shuffle(dataset_node)
            datasets_list.append(dataset_node)

        rest_dataset = []
        if final_index_t0 < len(data_t0):
            rest_dataset.append(data_t0[final_index_t0:, :])
        if final_index_t1 < len(data_t1):
            rest_dataset.append(data_t1[final_index_t1:, :])

        if len(rest_dataset) > 0:
            # rest_dataset = np.vstack(rest_dataset)
            warnings.warn('There are patients that were not included in the split')
            # datasets_list.append(rest_dataset)

        self.datasets = datasets_list
        self.datasets_train = datasets_list[:self.n_nodes]
        self.dataset_test = datasets_list[-2]
        self.dataset_validation = datasets_list[-1]

        # save in self.split_info the number of treated/untreated patients in each training dataset
        self.split_info = []
        for i in range(self.n_nodes):
            self.split_info.append([len(self.datasets_train[i][self.datasets_train[i][:, self.columns.index('t')] == 1]), len(self.datasets_train[
                                                                                                                                  i][
                                                                                                                                  self.datasets_train[i][:, self.columns.index('t')] == 0])])





    #
    def pad_datasets(self):
        '''
        Pad the datasets with zeros to have the same length to include as list in the fit method
        :return: None
        self.datasets has a list of datasets with the same length
        '''
        datasets_copy = copy.deepcopy(self.datasets)
        width = self.datasets[0].shape[1]
        max_len = max([len(self.datasets[i]) for i in range(len(self.datasets))])
        new_columns = copy.deepcopy(self.columns)

        for i, ds in enumerate(self.datasets):
            datasets_copy[i] = np.hstack((datasets_copy[i], np.ones(shape=(len(ds), 1))))
            if len(ds) != max_len:
                if len(ds) > max_len:
                    max_len = len(ds)

        for j, ds in enumerate(datasets_copy):
            difference = max_len - len(ds)
            aux = np.zeros(shape=(difference, width + 1))
            datasets_copy[j] = np.vstack((ds, aux))

        new_columns.append('flag')

        self.datasets = datasets_copy
        self.datasets_train = self.datasets[:self.n_nodes]
        self.dataset_test = self.datasets[-2]
        self.dataset_validation = self.datasets[-1]

    def prune_datasets(self):
        '''
        remove the padded zeros from the datasets
        :return: None
        '''

        datasets_copy = copy.deepcopy(self.datasets)

        for i, ds in enumerate(self.datasets):
            flag_col = ds[:, -1]
            mask = flag_col == 1
            datasets_copy[i] = ds[mask][:, :-1]# remove the last column of ds

        self.columns = self.columns[:-1]
        self.datasets = datasets_copy
        self.datasets_train = self.datasets[:self.n_nodes]
        self.dataset_test = self.datasets[-2]
        self.dataset_validation = self.datasets[-1]


    def scale_y_distributed(self, distributed: bool = True):
        '''
        Scale the y column of the datasets
        :param distributed: if distributed is true, the y_scaler is calculated as the weighted average of the scalers of each node; else, the y_scaler is calculated
        with all the samples of the training set (centralized)
        :return:
        '''
        y_train_list = [ds[:, self.columns.index('y')][..., np.newaxis] for ds in self.datasets_train]
        if distributed:
            y_train_scalers_list = []
            n_datapoints_list = []
            n_datapoints_total = 0
            for i, y in enumerate(y_train_list):
                y_scaler = StandardScaler()
                y_scaler.fit(y)
                n_datapoints_list.append(len(y))
                n_datapoints_total += len(y)
                y_train_scalers_list.append(y_scaler)

            mean_weighted_avg = 0
            std_weighted_avg = 0
            for i_scaler, y_scaler in enumerate(y_train_scalers_list):
                mean_weighted_avg += (y_scaler.mean_ * n_datapoints_list[i_scaler]) / n_datapoints_total
                std_weighted_avg += (y_scaler.scale_ * n_datapoints_list[i_scaler]) / n_datapoints_total

            weighted_scaler = StandardScaler()
            weighted_scaler.mean_ = mean_weighted_avg
            weighted_scaler.scale_ = std_weighted_avg

            self.y_scaler = weighted_scaler

        else:
            # concatenate y_train_list as unique vector
            y_train = np.concatenate(y_train_list)
            y_scaler = StandardScaler()
            y_scaler.fit(y_train)
            self.y_scaler = y_scaler

        for i in range(self.n_nodes):
            y_train_scaled = self.y_scaler.transform(y_train_list[i])
            self.datasets_train[i][:, self.columns.index('y')] = y_train_scaled.squeeze()

        y_val_scaled = self.y_scaler.transform(self.dataset_validation[:, self.columns.index('y')][..., np.newaxis])
        self.dataset_validation[:, self.columns.index('y')] = y_val_scaled.squeeze()

        y_test_scaled =self.y_scaler.transform(self.dataset_test[:, self.columns.index('y')][..., np.newaxis])
        self.dataset_test[:, self.columns.index('y')] = y_test_scaled.squeeze()


    def inverse_scaling_y(self):
        y_train_list = [ds[:, self.columns.index('y')][..., np.newaxis] for ds in self.datasets_train]
        for i in range(self.n_nodes):
            y_train = self.y_scaler.inverse_transform(y_train_list[i])
            self.datasets_train[i][:, self.columns.index('y')] = y_train.squeeze()

            y_val = self.y_scaler.inverse_transform(self.dataset_validation[:, self.columns.index('y')][..., np.newaxis])
            self.dataset_validation[:, self.columns.index('y')] = y_val.squeeze()

            y_test = self.y_scaler.inverse_transform(self.dataset_test[:, self.columns.index('y')][..., np.newaxis])
            self.dataset_test[:, self.columns.index('y')] = y_test.squeeze()

            dummy_scaler = StandardScaler()
            dummy_scaler.mean_ = 0
            dummy_scaler.scale_ = 1
            self.y_scaler = dummy_scaler



if __name__=="__main__":
    data_path = Path('.')
    # ihdp_catenets = Dataset(1, data_set='catenets_ihdp', data_path=data_path)
    # twins = Dataset(1, data_set='twins', data_path=data_path)
    # acic = Dataset(1, data_set='acic', data_path=data_path)
    # ihdp_tedvae = Dataset(1, data_set='tedvae_ihdp', data_path=data_path)

    ihdp = MultiNodeDataset(i_exp=1, data_set='tedvae_ihdp_a', data_path=data_path, train_test_val_split=[547, 100, 100],
                            treatment_split='51,222,51,222', preprocess=False)
    import pickle
    ihdp.pad_datasets()
    ihdp.scale_y_distributed()
    file = open('data_{}'.format(1), 'wb')
    pickle.dump(ihdp, file)
    file.close()

    a = 10

    file = open('data_{}'.format(1), 'rb')
    data = pickle.load(file)
    file.close()
