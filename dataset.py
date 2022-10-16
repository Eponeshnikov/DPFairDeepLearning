import os
import torch
import numpy as np
import pandas as pd
import urllib.request
from torch.utils.data import DataLoader
from utils import convert2torch, train_test_split
from data_processing import preprocessing_adult


class AdultDataset(torch.utils.data.Dataset):
    """ Create traning data iterator """

    def __init__(self, feature_X, label_y, sensetive_a):
        self.X = feature_X.float()
        self.y = label_y.float()
        self.A = sensetive_a.float()
        if type(self.A) == np.ndarray:
            self.A = torch.from_numpy(self.A).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.X[idx, :], self.y[idx], self.A[idx]


class Dataset:
    def __init__(self, args_):
        self.test_data_loader = None
        self.train_data_loader = None
        self.train_data = None
        self.test_data = None
        self.S_test = None
        self.S_train = None
        self.y_test = None
        self.y_train = None
        self.X_test = None
        self.X_train = None
        self.args = args_

    def dataset_preprocess(self):
        self.download_data()
        if self.args.dataset == 'Adult':
            if self.args.sensattr == 'age':
                use_age = True
            elif self.args.sensattr == 'sex':
                use_age = False
            else:
                raise Exception('Only age and sex are supported')
            X_train, y_train, S_train, X_test, y_test, S_test = \
                preprocessing_adult(self.args.data_dir, use_age=use_age,
                                    age_val=(self.args.age_low, self.args.age_high), seed=self.args.seed)
        else:
            raise Exception("Only Adult is available")
        X_train, X_test, y_train, y_test, S_train, S_test = convert2torch(X_train, X_test, y_train, y_test, S_train,
                                                                          S_test)
        self.X_train, self.X_test, self.y_train, self.y_test, self.S_train, self.S_test = \
            X_train, X_test, y_train, y_test, S_train, S_test

    def download_data(self):
        if not os.path.exists(self.args.data_dir):
            os.makedirs(self.args.data_dir)
        if 'Adult' in self.args.dataset:
            if not os.path.isfile(f'{self.args.data_dir}/adult.data'):
                print('Downloading adult.data ...')
                urllib.request.urlretrieve('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
                                           f'{self.args.data_dir}/adult.data')
                print('Downloaded')
            else:
                if self.args.only_download_data:
                    print('adult.data already downloaded')
            if not os.path.isfile(f'{self.args.data_dir}/adult.test'):
                print('Downloading adult.test ...')
                urllib.request.urlretrieve('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test',
                                           f'{self.args.data_dir}/adult.test')
                print('Downloaded')
            else:
                if self.args.only_download_data:
                    print('adult.test already downloaded')

    def get_dataset(self, return_=True):
        if any([self.X_train is None, self.X_test is None, self.y_train is None,
                self.y_test is None, self.S_train is None, self.S_test is None]):
            self.dataset_preprocess()
        self.train_data = AdultDataset(self.X_train, self.y_train, self.S_train)
        self.test_data = AdultDataset(self.X_test, self.y_test, self.S_test)
        if return_:
            return self.train_data, self.test_data

    def get_dataloader(self):
        if any([self.train_data is None, self.test_data is None]):
            self.get_dataset(return_=False)
        self.train_data_loader = DataLoader(self.train_data, batch_size=self.args.batch, shuffle=True)
        self.test_data_loader = DataLoader(self.test_data, batch_size=self.args.batch, shuffle=False)
        return self.train_data_loader, self.test_data_loader

    def n_features(self):
        if self.X_train is None:
            self.dataset_preprocess()
        return self.X_train.shape[1]

    def dataset_size(self):
        if self.X_train is None:
            self.dataset_preprocess()
        return self.X_train.shape[0]

    def n_classes(self):
        if self.y_train is None:
            self.dataset_preprocess()
        return len(np.unique(self.y_train))

    def n_groups(self):
        if self.S_train is None:
            self.dataset_preprocess()
        return len(np.unique(self.S_train))
