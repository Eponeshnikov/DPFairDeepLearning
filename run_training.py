import sys
import torch
import pandas as pd
import numpy as np
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
    raise Exception("Only German, Adult_1, Adult_2 are available")

X_train, X_test, y_train, y_test, S_train, S_test = convert2torch(X_train, X_test, y_train, y_test, S_train,
                                                                  S_test)

n_feature = X_train.shape[1]
latent_dim = 15  # latent dim space as in LAFTR

hidden_layers = {'class': 200, 'avd': 200, 'ae': 200}

# create dataset loader
train_data = DatasetLoader(X_train, y_train, S_train)
test_data = DatasetLoader(X_test, y_test, S_test)

has_gpu = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# batch size
batch_size = 512
epoch = 1000

DELTA = 1 / X_train.shape[0]
MAX_GRAD_NORMS = [200]
EPSILONS = [11.5, 3.2, 0.96, 0.72]

privacy_args = []
for EPSILON in EPSILONS:
    for MAX_GRAD_NORM in MAX_GRAD_NORMS:
        args = {"MAX_GRAD_NORM": MAX_GRAD_NORM, "EPSILON": EPSILON, "DELTA": DELTA}
        privacy_args.append(args)

parts_to_privacy = ['autoencoder', 'adversary', 'classifier']
comb_privacy = []

for n in range(0, 1):
    for i in itertools.combinations(parts_to_privacy, n):
        comb_privacy.append(i)

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

if len(sys.argv) > 2:
    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    lfr = DemParModel(n_feature=n_feature, latent_dim=latent_dim, class_weight=1, recon_weight=0,
                      adv_weight=1, hidden_layers=hidden_layers)
    trainer = Trainer(lfr, [train_data_loader, test_data_loader],
                      DATA_SET_NAME, "LFR")

    trainer.train_process(comb_privacy_eps[c_n][0], comb_privacy_eps[c_n][1], epoch)
    exit()
