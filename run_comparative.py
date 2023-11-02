import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import subprocess
import numpy as np
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')
import pickle


n_nodes = 2
hyperparams_path = 'federated/hyperparams'
selected_hp = [0]

data_settings = [ 'tedvae_ihdp','tedvae_ihdp_a']# possible values: tedvae_ihdp, tedvae_ihdp_a, acic, jobs, twins
train_test_val_splits = ['547,100,100', '547,100,100']
data_path = './datasets'
fedavg_strategies='fedavg_all,fedavg_all_vanilla,none'# possible values: fedavg_all, fedavg_reg, fedavg_reg_enc, fedavg_reg_dec, fedavg_enc, fedavg_dec, fedavg_enc_dec, none. Comma separated values (without spaces) for multiple strategies.
#all these fedavg_strategies are included to do an ablation study

datasets = np.arange(1,21)
verbose=1

fedci_bool = False
causalrff_bool = False
DEBUG = False

#load the dict with imbalances
with open('imbalances.pkl', 'rb') as f:
    imbalances_dict = pickle.load(f)

imbalances_names = ['tedvae_ihdp_a_random', 'tedvae_ihdp_a_ablation_vanilla']


python_file="federated/fed_tedvae.py"
python_presentation="present_results.py"

python_centralized_file = "centralized/centralized_tedvae.py"

python_fedci_file = "FedCI/fed_ci.py"

python_causalrff_file = "CausalRFF/causal_rff.py"

def run_script(arguments, resultspath):
    # arg1: selected_hp
    # arg2: fedavg_strategies
    # arg3: imbalance
    # arg4: dataset

    hp, fedavg, imb, dataset, d_setting, train_test_val_split= arguments

    # Modify the script parameters using the provided arguments
    # Update the necessary variables here with the modified values

    # Generate the command to run the Python script with modified arguments
    _resultspath = resultspath + '/' + str(imb)

    if not os.path.exists(_resultspath + f'/dset_{dataset}_{fedavg}_params_{hp}'):
        print(f'TRAINING: dataset {dataset}, imbalance {imb}, fedavg {fedavg}, selectedhp {hp}')
        command = ['python', python_file,
                   '--experiment', str(dataset),
                   '--nnodes', str(n_nodes),
                   '--verbose', str(verbose),
                   '--fedavg', fedavg,
                   '--datapath', data_path,
                   '--data_setting', d_setting,
                   '--hyperpath', hyperparams_path,
                   '--resultspath', _resultspath,
                   '--selectedhp', str(hp),
                   '--train_test_val_split', train_test_val_split,
                   '--treatment_split', str(imb),
                   '--debugger', str(DEBUG)]

        # Run the script with the modified arguments using subprocess
        subprocess.run(command)

def run_centralized(arguments, resultspath, verbose):

    imb, dataset, d_setting, train_test_val_split = arguments
    # arg1: imbalance

    # Generate the command to run the Python script with modified arguments
    _resultspath = resultspath + '/' + str(imb)

    if not os.path.exists(_resultspath + f'/dset_{dataset}_centralized'):
        print(f'TRAINING centralized: dataset {dataset}, imbalance {imb}')
        command = ['python', python_centralized_file,
                   '--experiment', str(dataset),
                   '--verbose', str(verbose),
                   '--datapath', data_path,
                   '--data_setting', d_setting,
                   '--resultspath', _resultspath,
                   '--train_test_val_split', train_test_val_split,
                   '--treatment_split', str(imb),
                   '--debugger', str(DEBUG)]
        # command += ['--splittreatment'] if splittreatment else []

        # Run the script with the modified arguments using subprocess
        subprocess.run(command)

def run_fedci(arguments, resultspath, verbose):
    imb, dataset, d_setting, train_test_val_split = arguments
    # arg1: imbalance
    # arg2: dataset
    # Generate the command to run the Python script with modified arguments
    _resultspath = resultspath + '/' + str(imb)

    if not os.path.exists(_resultspath + f'/dset_{dataset}_centralized'):
        print(f'TRAINING FedCI: dataset {dataset}, imbalance {imb}')
        command = ['python', python_fedci_file,
                   '--experiment', str(dataset),
                   '--verbose', str(verbose),
                   '--datapath', data_path,
                   '--data_setting', d_setting,
                   '--resultspath', _resultspath,
                   '--train_test_val_split', train_test_val_split,
                   '--treatment_split', str(imb)]
        # command += ['--splittreatment'] if splittreatment else []

        # Run the script with the modified arguments using subprocess
        subprocess.run(command)

def run_causalrff(arguments, resultspath, verbose):
    imb, dataset, d_setting, train_test_val_split = arguments
    # arg1: imbalance
    # arg2: dataset
    # Generate the command to run the Python script with modified arguments
    _resultspath = resultspath + '/' + str(imb)

    if not os.path.exists(_resultspath + f'/dset_{dataset}_centralized'):
        print(f'TRAINING FedCI: dataset {dataset}, imbalance {imb}')
        command = ['python', python_causalrff_file,
                   '--experiment', str(dataset),
                   '--verbose', str(verbose),
                   '--datapath', data_path,
                   '--data_setting', d_setting,
                   '--resultspath', _resultspath,
                   '--train_test_val_split', train_test_val_split,
                   '--treatment_split', str(imb)]

        # Run the script with the modified arguments using subprocess
        subprocess.run(command)

def run_presentation(arguments, datasets, resultspath):
    imb, fed, hp = arguments
    # arg1: imbalance
    # arg2: fedavg
    # arg3: selectedhp
    # arg5: dataset

    # Generate the command to run the Python script with modified arguments
    # _resultspath = resultspath + '/' + str(imb)

    exps = ' '.join(str(i) for i in datasets)
    command = ['python', python_presentation,
                    '--fedavg', fed,
                    '--resultspath', resultspath,
                    '--selectedhp', str(hp),
                    '--imbalance', imb,
                    '--experiments', exps,
               '--fedci_bool', str(fedci_bool),
                '--causalrff_bool', str(causalrff_bool)
               ]

    # Run the script with the modified arguments using subprocess
    subprocess.run(command)

for data_setting, train_test_val_split in zip(data_settings, train_test_val_splits):
    if not os.path.exists(f'results/{data_setting}'):
        os.makedirs(f'results/{data_setting}')
    for  imbalance_name in imbalances_names:
        if data_setting in imbalance_name:
            results_path = f'results/{data_setting}/{imbalance_name}'
            imbalances = imbalances_dict[imbalance_name]
            if not os.path.exists(results_path):
                os.makedirs(results_path)

            file = open(results_path + '/imbalances', 'wb')
            np.save(file, imbalances)
            file.close()


            if DEBUG:
                n_jobs =1
            else:
                if data_setting=='acic':
                    n_jobs = 10 # acic dataset is too big to run in parallel 20 threads in our machine
                else:
                    n_jobs = 20

            combinations = [(hp, fedavg, imb, dataset, data_setting, train_test_val_split)
                           for hp in selected_hp
                           for fedavg in fedavg_strategies.split(',')
                           for imb in imbalances
                           for dataset in datasets]

            #Create combinations for centralized
            centralized_combinations = [(imb, dataset, data_setting, train_test_val_split)
                                        for imb in imbalances
                                        for dataset in datasets
                                        ]

            fedci_combinations = [(imb, dataset, data_setting, train_test_val_split)
                                        for imb in imbalances
                                        for dataset in datasets]


            # # # # Run the script in parallel for each combination
            Parallel(n_jobs=n_jobs, verbose=1)(delayed(run_script)(args, results_path) for args in combinations)
            # #
            Parallel(n_jobs=n_jobs, verbose=1)(delayed(run_centralized)(args, results_path, verbose=0) for args in centralized_combinations)

            if fedci_bool:
                Parallel(n_jobs=n_jobs, verbose=1)(delayed(run_fedci)(args, results_path, verbose=0) for args in fedci_combinations)

            if causalrff_bool:
                n_jobs = n_jobs//4
                Parallel(n_jobs=n_jobs, verbose=1)(delayed(run_causalrff)(args, results_path, verbose=0) for args in fedci_combinations)


            print('\nDATA SETTING ', data_setting)
            run_presentation((imbalance_name, fedavg_strategies, selected_hp[0]), datasets, results_path)



