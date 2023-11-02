import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import pickle
import subprocess
import warnings
warnings.filterwarnings('ignore')

verbose=1

data_setting = 'tedvae_ihdp' # you can set tedvae_ihdp_b, the rest of datasets is not available in this example.
test_val_split = '547,100,100'
data_path = './datasets'
fedavg_strategies='fedavg_all,fedavg_all_vanilla,none' # none is TV isolated

n_nodes = 2

datasets = [1] # np.arange(1,21) #datasets used in the experiment
#load the dict with imbalances

with open('imbalances.pkl', 'rb') as f:
    imbalances_dict = pickle.load(f)

imbalance_name = 'tedvae_ihdp_a_random'  # other option is 'tedvae_ihdp_a_ablation_vanilla'. To add more imbalances, modify imbalances.pkl via
# imbalances.py scrit
imbalance = imbalances_dict[imbalance_name]  # string of the imbalance

with open('imbalances.pkl', 'rb') as f:
    imbalances_dict = pickle.load(f)

imbalance_name = 'tedvae_ihdp_a_random'  # other option is 'tedvae_ihdp_a_ablation_vanilla'. To add more imbalances, modify imbalances.pkl via
# imbalances.py scrit
imbalance = imbalances_dict[imbalance_name]  # string of the imbalance

if not os.path.exists(f'results/{data_setting}'):
    os.makedirs(f'results/{data_setting}')

results_path = f'results/{data_setting}/{imbalance_name}'
imbalances = imbalances_dict[imbalance_name]
if not os.path.exists(results_path):
    os.makedirs(results_path)

print(results_path)

file = open(results_path + '/imbalances', 'wb')
np.save(file, imbalances)
file.close()

# for imb in imbalances:
#     _results_path = results_path + '/' + str(imb)
#     if not os.path.exists(_results_path):
#         os.makedirs(_results_path)
#     for dataset in datasets:
#         for fedavg in fedavg_strategies.split(','):
#             command = ['python', "federated/fed_tedvae.py",
#                        '--experiment', str(dataset),
#                        '--nnodes', str(n_nodes),
#                        '--verbose', str(0),
#                        '--fedavg', fedavg,
#                        '--datapath', data_path,
#                        '--data_setting', data_setting,
#                        '--resultspath', _results_path,
#                        '--train_test_val_split', test_val_split,
#                        '--treatment_split', str(imb)]
#
#             # Run the script with the modified arguments using subprocess
#             subprocess.run(command)
#
#         command = ['python', "centralized/centralized_tedvae.py",
#                              '--experiment', str(dataset),
#                    '--verbose', str(verbose),
#                    '--datapath', data_path,
#                    '--data_setting', data_setting,
#                    '--resultspath', _results_path,
#                    '--train_test_val_split', test_val_split,
#                    '--treatment_split', str(imb)]
#         # command += ['--splittreatment'] if splittreatment else []
#
#         # Run the script with the modified arguments using subprocess
#         subprocess.run(command)



exps = ' '.join(str(i) for i in datasets)
command = ['python', "present_results.py"
                '--fedavg', fedavg_strategies,
                '--resultspath', results_path,
                '--imbalance', imbalance_name,
                '--experiments', exps]