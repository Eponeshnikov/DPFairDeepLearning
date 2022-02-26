import os
from threading import Thread
import pandas as pd
from utils import  train_test_split2
import itertools


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


DELTA = 1 / X_train.shape[0]
MAX_GRAD_NORMS = [1, 1e-2]
EPSILONS = [11.5, 3.2, 0.96, 0.72]

privacy_args = []
for EPSILON in EPSILONS:
    for MAX_GRAD_NORM in MAX_GRAD_NORMS:
        args = {"MAX_GRAD_NORM": MAX_GRAD_NORM, "EPSILON": EPSILON, "DELTA": DELTA}
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
parallel_threads = 4
thread_list = []
repeats = 5
for j in range(repeats):
    for i in range(len(comb_privacy_eps)):
        n_threads += 1
        thread_list.append(Thread(target=os.system, args=('run_train_german.py ' + str(i),)))

        if n_threads % parallel_threads == 0 or i * (j + 1) == repeats * (len(comb_privacy_eps) - 1):
            for thread in thread_list:
                print('thread start')
                thread.start()
            for thread in thread_list:
                thread.join()
                print('thread end')
            print('=' * 40)
            thread_list = []
            n_threads = 0
