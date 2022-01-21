import torch
from torch import nn, optim
import pandas as pd
from data_processing import preprocessing_german as preprocessing
import numpy as np
from torch.utils.data import DataLoader
from utils import Logger, train_test_split2, DatasetLoader

import matplotlib.pyplot as plt

from fairness_metrics import cross_val_fair_scores

from model import DemParModel, EqualOddModel, EqualOppModel
from trainer import Trainer

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
from helper import plot_results


def add_sensitive_attribute(X, S):
    return torch.cat((X, S.unsqueeze(1)), 1)


df = pd.read_csv('./preprocessing/adult.csv')
#
S = df['gender_ Male'].values
del df['gender_ Male']
X = df.drop('outcome_ >50K', axis=1).values
y = df['outcome_ >50K'].values


X_train, X_test, y_train, y_test, S_train, S_test = train_test_split2(
    X, y, S, test_size=0.3)
X_s = add_sensitive_attribute(X_train, S_train)
X = torch.from_numpy(X).double()
y = torch.from_numpy(y).double()
S = torch.from_numpy(S).double()

print(X_train.shape, X_s[0:5, 30])

n_feature = X.shape[1]
latent_dim = 8  # latent dim space as in LAFTR
DATA_SET_NAME = "Adult"
logger = Logger('AutoEncoder', DATA_SET_NAME)

# create dataset loader
training_data = DatasetLoader(X_train, y_train, S_train)
test_data = DatasetLoader(X_test, y_test, S_test)

has_gpu = torch.cuda.is_available()

X_s = add_sensitive_attribute(X_train, S_train)

n_feature = X.shape[1]
DATA_SET_NAME = "Adult"
logger = Logger('AutoEncoder', DATA_SET_NAME)

# create dataset loader
training_data = DatasetLoader(X_train, y_train, S_train)
test_data = DatasetLoader(X_test, y_test, S_test)

has_gpu = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# batch size
batch_size = 64

hidden_layers = {'class': 8, 'avd': 8, 'ae': 8}

data_loader = DataLoader(training_data, batch_size=64, shuffle=True)
test_data_loader = DataLoader(test_data, batch_size=64, shuffle=True)

lfr_u = DemParModel(n_feature=n_feature, latent_dim=latent_dim, class_weight=1, recon_weight=0, adv_weight=0, hidden_layers={'class': 20, 'ae': 20, 'avd': 20})

trainer = Trainer(lfr_u, data_loader, DATA_SET_NAME, "LFR")
trainer.train(50)

results = {}

kfold = KFold(n_splits=5)

# Train a Logistic Regression classifier without fairness constraints
clr = LogisticRegression(max_iter=1000)
acc_, dp_, eqodd_, eopp_ = cross_val_fair_scores(clr, X_test.cpu().detach(
).numpy(), y_test.cpu().detach().numpy(), kfold, S_test.cpu().detach().numpy())
results["LR"] = ([np.mean(acc_), np.mean(dp_), np.mean(eqodd_), np.mean(eopp_)], [
                 np.std(acc_), np.std(dp_), np.std(eqodd_), np.std(eopp_)])

X_transformed = lfr_u.transform(X_test.to(device))
clr = LogisticRegression(max_iter=1000)
acc_, dp_, eqodd_, eopp_ = cross_val_fair_scores(clr, X_transformed.cpu().detach(
).numpy(), y_test.cpu().detach().numpy(), kfold, S_test.cpu().detach().numpy())
scores_ = [np.mean(acc_), np.mean(dp_), np.mean(eqodd_), np.mean(eopp_)]
std_ = [np.std(acc_), np.std(dp_), np.std(eqodd_), np.std(eopp_)]
results[lfr_u.name] = (scores_, std_)


print(results)

plot_results(results)
