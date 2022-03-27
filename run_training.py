import sys
import torch
import pandas as pd
import use_cuda
from torch.utils.data import DataLoader
from utils import train_test_split, DatasetLoader, convert2torch
from data_processing import preprocessing_german, preprocessing_adult_1, preprocessing_adult_2
import itertools

from model import DemParModel
from trainer import Trainer

if len(sys.argv) > 2:
    c_n = int(sys.argv[2])
else:
    c_n = None

if len(sys.argv) > 4:
    seed = int(sys.argv[4])
else:
    seed = 0

if len(sys.argv) > 2:
    DATA_SET_NAME = sys.argv[1]
else:
    DATA_SET_NAME = 'Adult_1'
if DATA_SET_NAME == 'German':
    train_data = pd.read_csv("./dataset/german.data.csv")
    X, y, S, data = preprocessing_german(train_data)
    X_train, X_test, y_train, y_test, S_train, S_test = train_test_split(X, y, S, test_size=0.3)

elif DATA_SET_NAME == 'Adult_1':
    train_data = pd.read_csv("./dataset/adult.data.csv")
    test_data = pd.read_csv("./dataset/adult.test.csv")
    X_train, y_train, S_train, data = preprocessing_adult_1(train_data)
    X_test, y_test, S_test, data_ = preprocessing_adult_1(test_data)

elif DATA_SET_NAME == 'Adult_1_s':
    train_data = pd.read_csv("./dataset/adult.data.csv")
    X, y, S, data = preprocessing_adult_1(train_data)
    X_train, X_test, y_train, y_test, S_train, S_test = train_test_split(X, y, S, test_size=0.3)

elif DATA_SET_NAME == 'Adult_2':
    data = preprocessing_adult_2("income", "sex")
    S_train = data["attr_train"]
    S_test = data["attr_test"]
    X_train = data["x_train"]
    X_test = data["x_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
else:
    raise Exception("Only German, Adult_1, Adult_2, Adult_1_s are available")

X_train, X_test, y_train, y_test, S_train, S_test = convert2torch(X_train, X_test, y_train, y_test, S_train,
                                                                  S_test)
# ======================== Parameters ========================
# ------------------Architecture parameters-------------------
n_feature = X_train.shape[1]  # change in specific cases

latent_dim_base = 15
min_latent_coef = 0
max_latent_coef = 1

neurons_in_layer_base = 20
min_neuron_coef = 0
max_neuron_coef = 1
# ------------------------------------------------------------
# Calculated automatically, change in specific cases
latent_dims = [latent_dim_base * (10 ** i) for i in range(min_latent_coef, max_latent_coef)]  # latent dim space
hidden_layers_ = [{'class': neurons_in_layer_base * (10 ** i),
                   'avd': neurons_in_layer_base * (10 ** i),
                   'ae': neurons_in_layer_base * (10 ** i)} for i in range(min_neuron_coef, max_neuron_coef)]

# --------------------Learning parameters---------------------
adv_on_batchs = [1]
batch_size = 1024
epoch = 200
use_cuda = use_cuda.use_cuda  # see use_cuda.py
xavier_weights = True
# ------------------------------------------------------------
# Calculated automatically, change in specific cases
device_name = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
device = torch.device(device_name)

# ---------------------Privacy parameters---------------------
DELTA = 1 / X_train.shape[0]  # change in specific cases
MAX_GRAD_NORMS = [0.01]
EPSILONS = [0.72, 0.96, 3.2, 11.5]

parts_to_privacy = ['autoencoder', 'adversary', 'classifier']
max_num_of_private_components = 1
min_num_of_private_components = 0  # 0 - learning without privacy

use_opacus = True
# ------------------------------------------------------------
# ============================================================

# create dataset loader
train_data = DatasetLoader(X_train, y_train, S_train)
test_data = DatasetLoader(X_test, y_test, S_test)

# Generate combinations of architecture
comb_arch = []
for latent_dim in latent_dims:
    for adv_on_batch in adv_on_batchs:
        for hidden_layers in hidden_layers_:
            comb_arch.append([latent_dim, adv_on_batch, hidden_layers])

# Define one specific architecture
if len(sys.argv) > 3:
    architecture = comb_arch[int(sys.argv[3])]
else:
    architecture = comb_arch[0]

# Generate privacy parameters combinations
privacy_args = []
for EPSILON in EPSILONS:
    for MAX_GRAD_NORM in MAX_GRAD_NORMS:
        args = {"MAX_GRAD_NORM": MAX_GRAD_NORM, "EPSILON": EPSILON, "DELTA": DELTA}
        privacy_args.append(args)

# Generate combinations of private components
comb_privacy = []
for n in range(min_num_of_private_components, max_num_of_private_components + 1):
    for i in itertools.combinations(parts_to_privacy, n):
        comb_privacy.append(i)

# Generate combinations of private components with different privacy parameters combinations
comb_privacy_eps = []
no_privacy = True
for c in comb_privacy:
    for arg in privacy_args:
        if len(c) > 0:
            comb_privacy_eps.append([c, arg])
        else:
            if no_privacy:
                comb_privacy_eps.append([c, arg])
                no_privacy = False

# Run one specific training (c_n index in list with all possible combinations)
if len(sys.argv) > 2:
    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    lfr = DemParModel(n_feature=n_feature, latent_dim=architecture[0], class_weight=1, recon_weight=0,
                      adv_weight=1, hidden_layers=architecture[2])
    trainer = Trainer(lfr, [train_data_loader, test_data_loader],
                      DATA_SET_NAME, "DP", xavier_weights=xavier_weights)
    trainer.seed = seed

    trainer.train_process(comb_privacy_eps[c_n][0], comb_privacy_eps[c_n][1], architecture[1], epoch,
                          use_opacus=use_opacus)
    exit()
