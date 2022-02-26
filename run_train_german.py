import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from utils import train_test_split2, DatasetLoader
import itertools


from model import DemParModel
from trainer import Trainer

c_n = int(sys.argv[1])


df = pd.read_csv('./preprocessing/german.csv')
#
S = df['Sex_male'].values
del df['Sex_male']
X = df.drop('Risk_good', axis=1).values
y = df['Risk_good'].values

X_train, X_test, y_train, y_test, S_train, S_test = train_test_split2(
    X, y, S, test_size=0.3)
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).float()
S = torch.from_numpy(S).float()

n_feature = X.shape[1]
latent_dim = 15  # latent dim space as in LAFTR
hidden_layers = {'class': 20, 'ae': 20, 'avd': 20}
DATA_SET_NAME = "German"
# logger = Logger('AutoEncoder', DATA_SET_NAME)

# create dataset loader
train_data = DatasetLoader(X_train, y_train, S_train)
test_data = DatasetLoader(X_test, y_test, S_test)

has_gpu = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# batch size
batch_size = 64

DELTA = 1 / X_train.shape[0]
MAX_GRAD_NORM = 1e-2
EPSILONS = [11.5, 3.2, 0.96, 0.72]

privacy_args = []
for e in EPSILONS:
    args = {"MAX_GRAD_NORM": MAX_GRAD_NORM, "EPSILON": e, "DELTA": DELTA}
    privacy_args.append(args)

parts_to_privacy = ['autoencoder', 'adversary', 'classifier']
comb_privacy = []

for n in range(0, 2):
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



train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
lfr = DemParModel(n_feature=n_feature, latent_dim=latent_dim, class_weight=1, recon_weight=0,
                  adv_weight=1, hidden_layers=hidden_layers)
trainer = Trainer(lfr, [train_data_loader, test_data_loader],
                  DATA_SET_NAME, "LFR")
epoch = 5
print(comb_privacy_eps[c_n])
# trainer.train_process(comb_privacy_eps[c_n][0], comb_privacy_eps[c_n][1], epoch)
exit()
