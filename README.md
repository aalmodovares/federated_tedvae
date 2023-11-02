# Federated TEDVAE

This code implementation includes models of federated and Centralized TEDVAE, used for the paper: Federated learning for causal inference using deep generative disentangled models


The implementation is done in Tensorflow, with support of Tensorflow probability. The original version of TEDVAE [1] is done in Pyro (pytorch).

Please, install the requirements in requirements.txt

''''
pip install -r requirements.txt
''''

To run the simulation example, you can use the notebook: ihdp_b_example.ipynb

Check that results are stored in results/{data_setting}/{imbalance}. Where data settings included in this code are: IHDP setting B ('tedvae_ihdp') and IHDP setting A ('tedvae_ihdp_a'). You can include more imbalances by modifying the script imbalances.py


RESOURCES AND TIME CONSUMMING:
If you wanna run experiments for several datasets, it is recommended to parallelize the process using joblib. It is recommended to 
Complete the training and inference process for one dataset of IHDP takes about 3 minutes in CPU AMD Ryzen 9 5950X 16-Core Processor.
Each process takes about 3 GB of RAM memmory.


CITE AS:
@inproceedings{
almod{\'o}var2023federated,
title={Federated learning for causal inference using deep generative disentangled models},
author={Alejandro Almod{\'o}var and Juan Parras and Santiago Zazo},
booktitle={Deep Generative Models for Health Workshop NeurIPS 2023},
year={2023},
url={https://openreview.net/forum?id=r7qL5vM3Aa}
}

[1] W. Zhang, L. Liu, and J. Li, “Treatment effect estimation with disentangled latent factors,” in AAAI Conference on Artificial Intelligence, 2020.
