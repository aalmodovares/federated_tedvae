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
fedavg_strategies='fedavg_all,fedavg_all_vanilla,none' # fedavg_all: FedAvg on TEDVAE with propensity adaptation,
#fedavg_all_vanilla: FedAvg on TEDVAE vanilla, none is TV isolated
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

Code is prepared to do an ablation study of the federated modules in the whole TEDVAE model. Comparison methods: FedCI [2] and CausalRFF [3] are not included in this GitHUB.

If you need code to run FedCI or CausalRFF for comparison, or if you find it difficult to find more datasets on causal inference (ACIC2016, TWINS, JOBS) do not hesitate to contact me: 

alejandro.almodovar@upm.es



## CITE AS:
````
@article{almodovar2024propensity,
  title={Propensity Weighted federated learning for treatment effect estimation in distributed imbalanced environments},
  author={Almod{\'o}var, Alejandro and Parras, Juan and Zazo, Santiago},
  journal={Computers in Biology and Medicine},
  volume={178},
  pages={108779},
  year={2024},
  publisher={Elsevier}
}
````
[1] W. Zhang, L. Liu, and J. Li, “Treatment effect estimation with disentangled latent factors,” in AAAI Conference on Artificial Intelligence, 2020.
[2] V. Vo, T. N. Hoang, Y. Lee, and T.-Y. Leong, “Federated estimation of causal effects from
observational data,” 2021.
[3] T. V. Vo, A. Bhattacharyya, Y. Lee, and T.-Y. Leong, “An adaptive kernel approach to federated
learning of heterogeneous causal effects,” Advances in Neural Information Processing Systems,
vol. 35, pp. 24459–24473, 2022
