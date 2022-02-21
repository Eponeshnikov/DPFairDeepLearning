import os
import time
from threading import Thread
from multiprocessing import Process
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


def gen_privacy_name(privacy_parts, eps):
    name = 'privacy'
    pr_parts = ['autoencoder', 'adversary', 'classifier']
    eps_flag = True if len(privacy_parts) > 0 else False
    for p in pr_parts:
        name += ' '
        name += p
        name += '='
        name += 'True' if p in privacy_parts else 'False'
    if eps_flag:
        name += ' ε=' + str(eps)
    else:
        name += ' ε=∞'
    return name


df = pd.read_csv('./preprocessing/german.csv')
#
S = df['Sex_male'].values
del df['Sex_male']
X = df.drop('Risk_good', axis=1).values
y = df['Risk_good'].values

X_train, X_test, y_train, y_test, S_train, S_test = train_test_split2(
    X, y, S, test_size=0.3)

n_feature = X.shape[1]
latent_dim = 15  # latent dim space as in LAFTR
DATA_SET_NAME = "German"

# batch size
batch_size = 128

DELTA = 1 / X_train.shape[0]
MAX_GRAD_NORM = 1.2
EPSILONS = np.arange(5, 51, 5)

privacy_args = []
for e in EPSILONS:
    args = {'autoencoder': {"MAX_GRAD_NORM": MAX_GRAD_NORM, "EPSILON": e, "DELTA": DELTA},
            'adversary': {"MAX_GRAD_NORM": MAX_GRAD_NORM, "EPSILON": e, "DELTA": DELTA},
            'classifier': {"MAX_GRAD_NORM": MAX_GRAD_NORM, "EPSILON": e, "DELTA": DELTA}}
    privacy_args.append(args)

parts_to_privacy = ['autoencoder', 'adversary', 'classifier']
comb_privacy = []

for n in range(0, 2):
    for i in itertools.combinations(parts_to_privacy, n):
        comb_privacy.append(i)
# print(len(comb_privacy))
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

hidden_layers = {'class': 20, 'ae': 20, 'avd': 20}

n_threads = 0
thread_list = []
print(len(comb_privacy_eps))
for j in range(4):
    for i in range(len(comb_privacy_eps)):
        n_threads += 1
        thread_list.append(Thread(target=os.system, args=('run_train.py ' + str(i),)))
        if n_threads % 5 == 0 or i*j == (4-1)*(len(comb_privacy_eps)-1):
            for thread in thread_list:
                print('thread start')
                thread.start()
            for thread in thread_list:
                thread.join()
                print('thread end')
            print('='*40)
            thread_list = []
            n_threads = 0

