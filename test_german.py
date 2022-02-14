import time
from threading import Thread
import torch
import pandas as pd
from data_processing import preprocessing_german as preprocessing
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils import Logger, train_test_split2, DatasetLoader
import itertools

from model import DemParModel, EqualOddModel, EqualOppModel
from trainer import Trainer


def add_sensitive_attribute(X_, S_):
    return torch.cat((X_, S_.unsqueeze(1)), 1)


def train(trainer_, epochs_, privacy_parts, privacy_args):
    trainer_.train_process(privacy_parts, privacy_args, epochs_)


def gen_privacy_name(privacy_parts):
    name = 'privacy'
    pr_parts = ['autoencoder', 'adversary', 'classifier']
    for p in pr_parts:
        name += ' '
        name += p
        name += '='
        name += 'True' if p in privacy_parts else 'False'
    return name


df = pd.read_csv('./preprocessing/german.csv')
#
S = df['Sex_male'].values
del df['Sex_male']
X = df.drop('Risk_good', axis=1).values
y = df['Risk_good'].values

X_train, X_test, y_train, y_test, S_train, S_test = train_test_split2(
    X, y, S, test_size=0.3)
X = torch.from_numpy(X).double()
y = torch.from_numpy(y).double()
S = torch.from_numpy(S).double()

n_feature = X.shape[1]
latent_dim = 15  # latent dim space as in LAFTR
DATA_SET_NAME = "German"
# logger = Logger('AutoEncoder', DATA_SET_NAME)

# create dataset loader
train_data = DatasetLoader(X_train, y_train, S_train)
test_data = DatasetLoader(X_test, y_test, S_test)

has_gpu = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# batch size
batch_size = 64

parts_to_privacy = {'autoencoder': {"MAX_GRAD_NORM": 1.2, "EPSILON": 50, "DELTA": 1e-5},
                    'adversary': {"MAX_GRAD_NORM": 1.2, "EPSILON": 50, "DELTA": 1e-5},
                    'classifier': {"MAX_GRAD_NORM": 1.2, "EPSILON": 50, "DELTA": 1e-5}}
comb_privacy = []

for n in range(0, 4):
    for i in itertools.combinations(parts_to_privacy, n):
        comb_privacy.append(i)

hidden_layers = {'class': 20, 'ae': 20, 'avd': 20}

train_data_loaders = [DataLoader(train_data, batch_size=batch_size, shuffle=True) for i in range(len(comb_privacy))]
test_data_loaders = [DataLoader(test_data, batch_size=batch_size, shuffle=True) for i in range(len(comb_privacy))]
data_loaders = [i for i in zip(train_data_loaders, test_data_loaders)]

lfrs = [DemParModel(n_feature=n_feature, latent_dim=latent_dim, class_weight=1, recon_weight=0,
                    adv_weight=1, hidden_layers=hidden_layers) for i in range(len(comb_privacy))]

trainers = []
for data_loader, lfr, parts_ in zip(data_loaders, lfrs, comb_privacy):
    trainers.append(Trainer(lfr, data_loader, DATA_SET_NAME, "LFR", gen_privacy_name(parts_)))

epochs = [10 for i in range(len(comb_privacy))]
threads = [Thread(target=train, args=(t, e, p, parts_to_privacy)) for t, e, p in zip(trainers, epochs, comb_privacy)]

for i, thread in enumerate(threads):
    print('start thread', i)
    thread.start()
for i, thread in enumerate(threads):
    thread.join()
    print('end thread', i)
