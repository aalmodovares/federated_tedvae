# Federated TEDVAE

This code implementation includes models of federated and Centralized TEDVAE, used for the paper: Federated learning for causal inference using deep generative disentangled models


The implementation is done in Tensorflow, with support of Tensorflow probability. The original version of TEDVAE [1] is done in Pyro (pytorch).

Please, install the requirements in requirements.txt

````
pip install -r requirements.txt
````

To run the simulation example, you can use the notebook: ihdp_b_example.ipynb

To change the data setting: IHDP A/B, modify data_setting variable.
````
data_setting = 'tedvae_ihdp' # you can set tedvae_ihdp_b, the rest of datasets is not available in this example.
fedavg_strategies='fedavg_all,fedavg_all_vanilla,none' # FedAvg vanilla, FedAvg with propensity adaptation, none is TV isolated
````

Define datasets to run as a list of their indices:

````
datasets =  np.arange(1,21) #datasets used in the experiment
````
To ensure that the split of data are the same as used in our experiments, the splitted data are in a folder called 'presaved_data'. If you want to make other split, remove the folders that you want.

Check that results are stored in results/{data_setting}/{imbalance}. Where data settings included in this code are: IHDP setting B ('tedvae_ihdp') and IHDP setting A ('tedvae_ihdp_a'). You can include more imbalances by modifying the script imbalances.py


## RESOURCES AND TIME CONSUMMING:

If you wanna run experiments for several datasets, it is recommended to parallelize the process using joblib: an example of the whole code is in run_comparative.py

````
python run_comparative.py
````

Complete the training and inference process for one dataset of IHDP takes about 3 minutes in CPU AMD Ryzen 9 5950X 16-Core Processor.
Each process takes about 3 GB of RAM memmory.

## DATA:

Datasets included are the same as in TEDVAE repository. 100 replications of each setting of IHDP. 

## EXTRA RESOURCES

If you need code to run FedCI or CausalRFF for comparison, or if you find it difficult to find more datasets on causal inference (ACIC2016, TWINS, JOBS) do not hesitate to contact me: 

alejandro.almodovar@upm.es



## CITE AS:
````
@inproceedings{
almod{\'o}var2023federated,
title={Federated learning for causal inference using deep generative disentangled models},
author={Alejandro Almod{\'o}var and Juan Parras and Santiago Zazo},
booktitle={Deep Generative Models for Health Workshop NeurIPS 2023},
year={2023},
url={https://openreview.net/forum?id=r7qL5vM3Aa}
}
````
[1] W. Zhang, L. Liu, and J. Li, “Treatment effect estimation with disentangled latent factors,” in AAAI Conference on Artificial Intelligence, 2020.
