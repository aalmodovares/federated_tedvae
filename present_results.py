'''to present results in ablation study'''

'''in this version, the split information about treated/untreated is presented in this way for each domain:
- domain_1: treated/untreated
- domain_2: treated/untreated, and so on '''
import warnings
warnings.filterwarnings("ignore")

import os
import sys
sys.path.insert(0, os.getcwd())
import argparse
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn
from pathlib import Path
from tabulate import tabulate
import pickle
import seaborn as sns

import re


if 'federated_tedvae/federated' in os.getcwd():
    os.chdir('..')


class Tee:
    def __init__(self, *outputs):
        self.outputs = outputs

    def write(self, text):
        for output in self.outputs:
            output.write(text)

    def flush(self):
        for output in self.outputs:
            output.flush()



from common_utils.plots import  get_one_dict_from_list, get_df_from_unique_key_dict

parser = argparse.ArgumentParser()
parser.add_argument("--fedavg", "-fs", default='fedavg_all,none', type=str, help='string of federated strategies, separated by ,')
parser.add_argument("--nnodes", "-n", default=2, type=int, help='Number of nodes')
parser.add_argument("--resultspath", "-rp", default='results/tedvae_ihdp', type=str, help='Results path')
parser.add_argument("--experiments", "-exps", nargs='+', default=1, type=str, help='array of all datasets to plot distributions')
parser.add_argument("--hyperpath", "-hpp", default='federated/hyperparams', type=str, help='Hyperparameters dataframe path')
parser.add_argument("--selectedhp", "-shp", nargs='+', default=[0], type=int, help='Array of selected hyperparams')
parser.add_argument("--imbalance", "-imb", default='fixed_imbalance_nonconstant', type=str, help='imbalance name to plot')
parser.add_argument('--fedci_bool', '-fedci', default='False', type=str, help='Boolean to plot FedCI results')
parser.add_argument('--causalrff_bool', '-causalrff', default='False', type=str, help='Boolean to plot CausalRFF results')

args = parser.parse_args()

path = Path(args.resultspath)

print_all = False
plot_all = False
print_latex = False


# Open a file for writing
output_file = open(path / 'output.txt', 'w')

# Create a Tee instance to combine console output and file output
tee = Tee(sys.stdout, output_file)

# Redirect stdout to the Tee instance
sys.stdout = tee



FEDERATED_STRATEGIES = args.fedavg.split(',')
hyperparams = pd.read_csv(Path(args.hyperpath) / 'hyperparams')
experiments = np.array(args.experiments[0].split(), dtype=int)
selected_hp = np.array(args.selectedhp)[..., np.newaxis]
n_nodes = args.nnodes

file_imbalances = open('imbalances.pkl', 'rb')
imbalances = pickle.load(file_imbalances)
file_imbalances.close()

imbalances = imbalances[args.imbalance]

PLOT_HISTORY = False

FED_CI = args.fedci_bool=='True'
CAUSAL_RFF = args.causalrff_bool=='True'

# experiments = np.arange(1,21)
df_list = []
results_imbalance = dict()

# create a dict with results
for imbalance in imbalances:
    results_list = []
    pehe_train_centralized = []
    pehe_test_centralized = []
    if FED_CI:
        pehe_train_fedci = []
        pehe_test_fedci = []
        n_dsets_fedci = 0

    if CAUSAL_RFF:
        pehe_train_causalrff = []
        pehe_test_causalrff = []
        n_dsets_causalrff = 0

    results_path = path / str(imbalance)

    for i_experiment in experiments:
        i_exp = i_experiment
        results = dict()
        # for i_param, params in hyperparams.iterrows():
        #     if i_param in selected_hp:

        i_param = selected_hp[0][0]
        results[f'params_{i_param}'] = dict()
        for federated_strategy in FEDERATED_STRATEGIES:
            results[f'params_{i_param}'][federated_strategy] = dict()
            file = open(results_path / f'dict_dset_{i_exp}_{federated_strategy}_params_{i_param}', 'rb')
            res_dict = pickle.load(file)
            file.close()

            results[f'params_{i_param}'][federated_strategy]['pehe_train'] = res_dict['pehe_train']
            if print_all:
                print(f'PEHE TRAIN  - {federated_strategy.upper()} dset {i_exp + 1}')
                print('domain 1 / domain 2')
                print(res_dict['pehe_train'])

            results[f'params_{i_param}'][federated_strategy]['pehe_test'] = res_dict['pehe_test']
            if print_all:
                print(f'PEHE TEST  - {federated_strategy.upper()} dset {i_exp + 1}')
                print('domain 1 / domain 2')
                print(res_dict['pehe_test'])

            if PLOT_HISTORY:
                model_history = res_dict['model_history']
                prediction_columns = ['loss_p_t', 'loss_p_y', 'val_loss_p_t', 'val_loss_p_y']
                prediction_columns = [f'Domain[{i}] {loss}' for loss in prediction_columns for i in range(n_nodes)]

                prediction_history = model_history.loc[:, prediction_columns]
                prediction_history.plot(logy=True)
                plt.title(f'{federated_strategy.upper()} dset {i_exp + 1} imb {imbalance} params')
                plt.savefig(path / f'losses_log_dset_{i_exp}_{federated_strategy}_params_{i_param}')
                if plot_all:
                    plt.show()
                plt.close()


        file = open(results_path / f'dict_dset_{i_exp}_centralized', 'rb')
        centralized_dict = pickle.load(file)
        file.close()
        pehe_train_centralized.append(centralized_dict['pehe_train'])
        pehe_test_centralized.append(centralized_dict['pehe_test'])

        if FED_CI:
            if os.path.exists(results_path / f'dict_dset_{i_exp}_fedci'):
                # FedCI
                n_dsets_fedci+=1
                file_fedci = open(results_path / f'dict_dset_{i_exp}_fedci', 'rb')
                fedci_dict = pickle.load(file_fedci)
                file_fedci.close()
                pehe_train_fedci.append(fedci_dict['pehe_train'])
                pehe_test_fedci.append(fedci_dict['pehe_test'])

        if CAUSAL_RFF:
            if os.path.exists(results_path / f'dict_dset_{i_exp}_causalrff'):

                # FedCI
                n_dsets_causalrff+=1
                file_causalrff = open(results_path / f'dict_dset_{i_exp}_causalrff', 'rb')
                causalrff_dict = pickle.load(file_causalrff)
                file_causalrff.close()
                pehe_train_causalrff.append(causalrff_dict['pehe_train'])
                pehe_test_causalrff.append(causalrff_dict['pehe_test'])



        results_list.append(results)

    results_imbalance[f'imbalance_{imbalance}'] = results_list


    n_dsets = i_exp
    df_train_centralized = pd.DataFrame(np.array([['train']*n_dsets, ['none']*n_dsets,  ['centralized']*n_dsets, pehe_train_centralized]).T, columns=['train', 'domain', 'fed_case', 'PEHE'])
    df_test_centralized = pd.DataFrame(np.array([['test']*n_dsets, ['none']*n_dsets,  ['centralized']*n_dsets, pehe_test_centralized]).T,
                                        columns=['train', 'domain', 'fed_case', 'PEHE'])
    df_centralized = pd.concat((df_train_centralized, df_test_centralized), axis=0, ignore_index=True)

    # print(f'error imbalance: {imbalance}')
    _, flattened_dict, unique_key_dict = get_one_dict_from_list(results_list)
    df = get_df_from_unique_key_dict(unique_key_dict)
    df = pd.concat((df_centralized, df), axis=0, ignore_index=True)

    results_imbalance[f'imbalance_{imbalance}_df'] = df

    df['imbalance'] = imbalance

    df_list.append(df)

    if FED_CI:
        # FedCI
        pehe_train_array = np.array(pehe_train_fedci) # [:, i] is for i-th domain
        pehe_test_array = np.array(pehe_test_fedci)
        for n in range(n_nodes):
            df_train_fedci = pd.DataFrame(np.array([['train'] * n_dsets_fedci, [n+1] * n_dsets_fedci, ['fedci'] * n_dsets_fedci,
                                                    pehe_train_array[:, n]]).T,
                         columns=['train', 'domain', 'fed_case', 'PEHE'])
            df_test_fedci = pd.DataFrame(np.array([['test'] * n_dsets_fedci, [n+1] * n_dsets_fedci, ['fedci'] * n_dsets_fedci,
                                                   pehe_test_array[:, n]]).T,
                            columns=['train', 'domain', 'fed_case', 'PEHE'])
            df_fedci = pd.concat((df_train_fedci, df_test_fedci), axis=0, ignore_index=True)
            df_fedci['imbalance'] = imbalance
            df_fedci['domain'] = df_fedci['domain'].astype('int32')
            df_list.append(df_fedci)

    if CAUSAL_RFF:
        pehe_train_array = np.array(pehe_train_causalrff)  # [:, i] is for i-th domain
        pehe_test_array = np.array(pehe_test_causalrff)
        for n in range(n_nodes):
            df_train_causalrff = pd.DataFrame(np.array([['train'] * n_dsets_causalrff, [n + 1] * n_dsets_causalrff, ['causalrff'] * n_dsets_causalrff,
                                                    pehe_train_array[:, n]]).T,
                                          columns=['train', 'domain', 'fed_case', 'PEHE'])
            df_test_causalrff = pd.DataFrame(np.array([['test'] * n_dsets_causalrff, [n + 1] * n_dsets_causalrff, ['causalrff'] * n_dsets_causalrff,
                                                   pehe_test_array[:, n]]).T,
                                         columns=['train', 'domain', 'fed_case', 'PEHE'])
            df_causalrff = pd.concat((df_train_causalrff, df_test_causalrff), axis=0, ignore_index=True)
            df_causalrff['imbalance'] = imbalance
            df_causalrff['domain'] = df_causalrff['domain'].astype('int32')
            df_list.append(df_causalrff)




patterns = {
    'TV FedAvg': r'.*all(?!_vanilla).*',
    'TV FedAvg Vanilla': r'.*all_vanilla.*',
    'fedavg_reg_enc': r'.*reg_enc(?!_dec).*',
    'fedavg_reg_dec': r'.*reg_dec(?!_enc).*',
    'fedavg_reg': r'.*reg(?!_enc|_dec).*',
    'fedavg_enc': r'(?<!reg_)enc(?!_dec).*',
    'fedavg_dec': r'(?<!reg_|enc_)dec.*',
    'fedavg_enc_dec': r'(?<!reg_)enc_dec.*',
    'TV Cen': r'.*centralized.*',
    'TV Iso': r'.*none.*',
    'FedCI': r'.*fedci.*',
    'CausalRFF': r'.*causalrff.*',
}

# Function to extract the desired pattern from a string
def extract_pattern(s):
    found_pattern = None
    for pattern_name, pattern in patterns.items():
        if re.search(pattern, s):
            found_pattern = pattern_name
            break
    if found_pattern is None:
        found_pattern = 'isolated'
    return found_pattern


df = pd.concat(df_list, axis=0, ignore_index=True)

# Apply the function to the DataFrame column
df['Setting'] = df['fed_case'].apply(extract_pattern)

#convert ITE dtype to float32
df['PEHE'] = df['PEHE'].astype('float32')

#Get the unique values of fed_case
fed_cases = df['Setting'].unique()


#plot mean of errors for each domain. X axis is the imbalance level and the hue is the train column. Use fill between to plot the std
#define a figure to make one subplot for each domain
fig, axes = plt.subplots(nrows=n_nodes, ncols=1, figsize=(10, 7 * n_nodes))
fig_test, axes_test = plt.subplots(nrows=n_nodes, ncols=1, figsize=(10, 7 * n_nodes))

# Define color palettes for train and test plots
train_palette = sns.color_palette('tab10',22)
# Define line styles for each variable
line_styles = ['-', '--', '-.', ':']
n_cases = len(df['Setting'].unique())

# Initialize an empty DataFrame to store the statistics
statistics_list = []

df['imbalance_list'] = df['imbalance'].apply(lambda x: x.split(','))
df['imbalance_list'] = df.apply(lambda row: [f"{row['imbalance_list'][i]}/{row['imbalance_list'][i + 1]}" for i in
                                                                    range(0, len(row['imbalance_list']), 2)], axis=1)
# Transform the column into a single string with newline separation
df['imbalance_list'] = df['imbalance_list'].str.join('\n')

for j in range(n_nodes):
    train_data = df[((df['domain'] == (j + 1)) + (df['domain'] == 'none')) & (df['train'] == 'train')]
    test_data = df[((df['domain'] == (j + 1)) + (df['domain'] == 'none')) & (df['train'] == 'test')]

    # Plot train data with train_palette colors and add train labels to the legend
    sns.lineplot(x='imbalance_list', y='PEHE', hue='Setting', data=train_data, ax=axes[j], palette=train_palette[:n_cases],
                 style='Setting', markers=True, dashes=False,
                 errorbar=('pi', 50),
                 err_kws={'alpha': 0.05})


    # Calculate and store the statistics for train data
    for variable in train_data['Setting'].unique():

        for imb in train_data['imbalance'].unique():
            train_variable_data = train_data[(train_data['Setting'] == variable) & (train_data['imbalance'] == imb)]['PEHE']
            median = np.median(train_variable_data)
            percentile_25 = np.percentile(train_variable_data, 25)
            percentile_75 = np.percentile(train_variable_data, 75)
            mean = np.mean(train_variable_data)
            std = np.std(train_variable_data)
            aux_df = pd.DataFrame({'domain': j + 1, 'Setting': variable, 'imbalance':imb, 'train/test': 'train', 'median': median,
                                 'percentile_25': percentile_25, 'percentile_75': percentile_75,
                                   'mean': mean, 'std':std}, index=[0])

            statistics_list.append(aux_df)


    # Plot test data with test_palette colors and add test labels to the legend
    sns.lineplot(x='imbalance_list', y='PEHE', hue='Setting', data=test_data, ax=axes[j], palette=train_palette[n_cases:],
                 style='Setting', markers=False, dashes=True,
                 errorbar=('pi', 50),
                 err_kws={'alpha': 0.05})

    # Plot test data alone in other figure
    sns.lineplot(x='imbalance_list', y='PEHE', hue='Setting', data=test_data, ax=axes_test[j], palette=train_palette[n_cases:],
                 style='Setting', markers=False, dashes=True,
                 errorbar=('pi', 50),
                 err_kws={'alpha': 0.05})


    test_statistics_list = []
    # Calculate and store the statistics for test data
    for variable in test_data['Setting'].unique():
        for imb in test_data['imbalance'].unique():
            test_variable_data = test_data[(test_data['Setting'] == variable) & (test_data['imbalance'] == imb)]['PEHE']
            median = np.median(test_variable_data)
            percentile_25 = np.percentile(test_variable_data, 25)
            percentile_75 = np.percentile(test_variable_data, 75)
            mean = np.mean(test_variable_data)
            std = np.std(test_variable_data)
            test_df = pd.DataFrame({'domain': j + 1, 'Setting': variable, 'imbalance':imb, 'train/test': 'test', 'median': median,
                                   'percentile_25': percentile_25, 'percentile_75': percentile_75,
                                    'mean': mean, 'std': std}, index=[0])

            statistics_list.append(test_df)
            test_statistics_list.append(test_df)

    # Set the x-axis label with the corresponding imbalance, stars for the corresponding domain
    # Get the current tick labels of the specific subplot
    xtick_labels = axes[j].get_xticklabels()


    # Modify the tick labels
    for label in xtick_labels:
        text = label.get_text()
        chains = text.split('\n')
        chains[j] = f'*{chains[j]}*'  # Modifying text style
        modified_text = '\n'.join(chains)
        label.set_text(modified_text)

    # Set the modified labels
    axes[j].set_xticklabels(xtick_labels)

    axes[j].set_title(f'domain: {j + 1}')
    handles, labels = axes[j].get_legend_handles_labels()

    # Modify the labels to include 'train' or 'test' prefix
    modified_labels = []
    for i_l, label in enumerate(labels):
        if i_l<n_cases:
            modified_labels.append('train: ' + label)
        else:
            modified_labels.append('test: ' + label)

    axes[j].legend(handles, modified_labels)
    # Set x-axis tick labels orientation to 45 degrees
    # axes[j].set_xticklabels(nodes_imbalances[j], rotation=45)
    axes[j].set_xlabel('treated/untreated in each domain')

    xtick_labels = axes_test[j].get_xticklabels()

    # Modify the tick labels
    for label in xtick_labels:
        text = label.get_text()
        chains = text.split('\n')
        chains[j] = f'*{chains[j]}*'  # Modifying text style
        modified_text = '\n'.join(chains)
        label.set_text(modified_text)

    # Set the modified labels
    axes_test[j].set_xticklabels(xtick_labels)

    axes_test[j].set_title(f'domain: {j + 1}')
    axes_test[j].set_xlabel('treated/untreated in each domain')

fig.suptitle('RESULTS')
fig.savefig(path /'ITEerr_lines.png')
if plot_all:
    fig.show()
plt.close(fig)

fig_test.suptitle('OUT OF SAMPLE RESULTS')
fig_test.savefig(path /'ITEerr_lines_test.png')
if plot_all:
    fig_test.show()
plt.close(fig_test)


# Print the statistics table
statistics_df = pd.concat((statistics_list), axis=0, ignore_index=True)
statistics_df['median(P25, P75)'] = statistics_df.apply(
            lambda row: f"{row['median']:.2f}({row['percentile_25']:.2f}, {row['percentile_75']:.2f})", axis=1)
statistics_df['mean(std)'] = statistics_df.apply(
            lambda row: f"{row['mean']:.2f}({row['std']:.2f})", axis=1)
statistics_df = statistics_df.drop(columns=['median', 'percentile_25', 'percentile_75'])
statistics_df = statistics_df.drop(columns=['mean', 'std'])



list_of_imbalances = list(df['imbalance'].unique())

# Extracting the values for domain 1 and domain 2 from the strings
# Extracting the values for domain 1 and domain 2 from the strings
treated_untreated = dict()
for n in range(n_nodes):
    treated_untreated['domain '+str(n+1)] = dict()
    treated_untreated['domain ' + str(n+1)]['treated'] = []
    treated_untreated['domain ' + str(n+1)]['untreated'] = []
    for item in list_of_imbalances:
        if item.split(',')[2*n+1] == 'random':
            treated_untreated['domain ' + str(n+1)]['treated'].append(int(item.split(',')[2 * n]))
            treated_untreated['domain ' + str(n+1)]['untreated'].append(int(item.split(',')[2 * n]))
        else:
            treated_untreated['domain '+str(n+1)]['treated'].append(int(item.split(',')[2*n]))
            treated_untreated['domain '+str(n+1)]['untreated'].append(int(item.split(',')[2*n+1]))

# Creating the figure and subplots
fig_imb, axs_imb = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

# Plotting the first subplot for domain 1
xticks = np.arange(len(list_of_imbalances))
width = 0.4
axs_imb[0].bar((xticks - width/2), treated_untreated['domain 1']['treated'], width=width, label='treated')
axs_imb[0].bar((xticks + width/2), treated_untreated['domain 1']['untreated'], width=width, label='untreated')
axs_imb[0].set_xticks(xticks)
axs_imb[0].set_xticklabels(xticks)
axs_imb[0].legend(loc='upper right')
axs_imb[0].set_title('Domain 1')
axs_imb[0].set_xlabel('treated/untreated in each domain, imbalance level')
axs_imb[0].set_ylabel('number of patients')

# Add horizontal lines at the first treated and untreated values of domain 1
axs_imb[0].axhline(y=treated_untreated['domain 1']['treated'][0], color='red', linestyle='--')
axs_imb[0].axhline(y=treated_untreated['domain 1']['untreated'][0], color='blue', linestyle='--')
axs_imb[0].text(-0.15, treated_untreated['domain 1']['treated'][0]+7.5, str(treated_untreated['domain 1']['treated'][0]), ha='right', va='center', color='red')
axs_imb[0].text(-0.15, treated_untreated['domain 1']['untreated'][0]+7.5, str(treated_untreated['domain 1']['untreated'][0]), ha='right', va='center',
                color='blue')

# Plotting the second subplot for domain 2
axs_imb[1].bar((xticks - width/2), treated_untreated['domain 2']['treated'], width=width, label='treated')
axs_imb[1].bar((xticks + width/2), treated_untreated['domain 2']['untreated'], width=width, label='untreated')
axs_imb[1].set_xticks(xticks)
axs_imb[1].set_xticklabels(xticks)
axs_imb[1].legend(loc='upper right')
axs_imb[1].set_title('Domain 2')
axs_imb[1].set_xlabel('treated/untreated in each domain, imbalance level')

axs_imb[1].axhline(y=treated_untreated['domain 2']['treated'][0], color='red', linestyle='--')
axs_imb[1].axhline(y=treated_untreated['domain 2']['untreated'][0], color='blue', linestyle='--')
axs_imb[1].text(-0.15, treated_untreated['domain 2']['treated'][0]+7.5, str(treated_untreated['domain 2']['treated'][0]), ha='right', va='center', color='red')
axs_imb[1].text(-0.15, treated_untreated['domain 2']['untreated'][0]+7.5, str(treated_untreated['domain 2']['untreated'][0]), ha='right',
                va='center', color='blue')


# Adding labels and adjusting layout
plt.suptitle('Treatment Distribution')
plt.tight_layout()

# Displaying the figure
plt.savefig(path /'imbalances.png')
if plot_all:
    plt.show()
plt.close()
# TABLE 3. Only for a selected imbalance level, show statistics for each domain
# Filter imbalance levels to be displayed
displayed_imbalances = imbalances
# displayed_imbalances = ['51,222,51,222','11,262,91,182']
# Create a train dataframe
train_statistics_df = statistics_df [statistics_df['train/test'] == 'train'].drop('train/test', axis=1)
#Create a test dataframe
test_statistics_df = statistics_df[statistics_df ['train/test'] == 'test'].drop('train/test', axis=1)
# create multiindex train and display only the selected imbalance level
#in statistics_df change train by 'In-sample' and test by 'Out-of-sample'
statistics_df['train/test'] = statistics_df['train/test'].apply(lambda x: 'In-sample' if x == 'train' else 'Out-of-sample')
multiindex_df = statistics_df.pivot(index=['train/test','domain', 'imbalance'], columns='Setting').T



multiindex_train_df = train_statistics_df.pivot(index=['domain', 'imbalance'], columns='Setting')
multiindex_train_df = multiindex_train_df.loc[(slice(None), displayed_imbalances), :]


# Create multiindex test and display only the selected imbalance level
multiindex_test_df = test_statistics_df.pivot(index=['domain', 'imbalance'], columns='Setting')
multiindex_test_df = multiindex_test_df.loc[(slice(None), displayed_imbalances), :]
# show the tables
df_train = multiindex_train_df.T
df_test = multiindex_test_df.T
# Adjust display options
pd.set_option('display.max_columns', None)  # Display all columns
pd.set_option('display.expand_frame_repr', False)  # Do not wrap the DataFrame
pd.set_option('display.max_colwidth', None)  # Display long strings without truncation
if print_all:
    print('IN SAMPLE RESULTS')
    print(df_train)
    print('OUT OF SAMPLE RESULTS')
    print(df_test)
    print('\n\n')

#FOR LATEX

#for multiindex with domain 1, keep only the two first terms of imbalance.split(',')
#for multiindex with domain 2, keep only the two last terms of imbalance.split(',')

# Modify the MultiIndex based on your criteria

df_train.columns = pd.MultiIndex.from_frame(pd.DataFrame([[domain, '/'.join(imbalance.split(",")[(domain-1)*2:((domain-1)*2+2)])] for domain,
imbalance in df_train.columns], columns=['domain', 'imbalance']))

df_train_median = df_train.loc['median(P25, P75)']
# df_train_median = df_train_median.reset_index()
df_train_mean = df_train.loc['mean(std)']
# df_train_mean = df_train_mean.reset_index()
if print_latex:
    print(f'IN SAMPLE RESULTS - {args.imbalance}\n\n')
    print('MEDIAN (p25/p75)\n\n')
    print(df_train_median.to_latex())
    print('MEAN (std)\n\n')
    print(df_train_mean.to_latex())


df_test.columns = pd.MultiIndex.from_frame(pd.DataFrame([[domain, '/'.join(imbalance.split(",")[(domain-1)*2:((domain-1)*2+2)])] for domain,
imbalance in df_test.columns], columns=['domain', 'imbalance']))

df_test_median = df_test.loc['median(P25, P75)']
# df_test_median = df_test_median.reset_index()
df_test_mean = df_test.loc['mean(std)']
# df_test_mean = df_test_mean.reset_index()

if print_latex:
    print(f'OUT OF SAMPLE RESULTS - {args.imbalance}\n\n')
    print('MEDIAN (p25/p75)\n\n')
    print(df_test_median.to_latex())
    print('MEAN (std)\n\n')
    print(df_test_mean.to_latex())


multiindex_df.columns = pd.MultiIndex.from_frame(pd.DataFrame([[tr, domain, '/'.join(imbalance.split(",")[(domain-1)*2:((domain-1)*2+2)])] for
                                                               tr, domain,
imbalance in multiindex_df.columns], columns=['train/test','domain', 'imbalance']))

#display in alphabetical order by rows of Setting in multiindex_df. _names of index: [None, 'Setting']
multiindex_df = multiindex_df.sort_index(level=[0,1], ascending=[True, True])
if print_latex:
    print('\n\n')
    print(multiindex_df)
if print_latex:
    print('\n\n')
    print(multiindex_df.to_latex())
df_mean = multiindex_df.loc['mean(std)']
print(df_mean)
if print_latex:
    print('\n\n')
    print(df_mean.to_latex())
    #print only out of sample results
    print('\n\n')

    print(df_mean.loc[:, 'Out-of-sample'].to_latex())

# Reset stdout to the console
sys.stdout = sys.__stdout__

# Close the output file
output_file.close()