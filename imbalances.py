# Description: This file contains the imbalances used in the experiments.

import pickle


ablation_vanilla = ['51,222,51,222', '11,262,91,182','1,272,101,172', '0,273,102,171'] # increasing imbalance

tedvae_ihdp_random = ['83,random,83,random'] # random is equivalent to sample with the original treatment assignment
imbalances = {
    'tedvae_ihdp_a_ablation_vanilla': ablation_vanilla,
    'tedvae_ihdp_a_random': tedvae_ihdp_random,
}

# save the dict
with open('imbalances.pkl', 'wb') as f:
    pickle.dump(imbalances, f)

