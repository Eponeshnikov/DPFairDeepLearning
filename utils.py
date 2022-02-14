import os
import numpy as np
import errno
from IPython import display
from matplotlib import pyplot as plt
import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter

'''
    TensorBoard Data will be stored in './runs' path
'''


class Logger:

    def __init__(self, model_name, data_name, privacy_name):
        self.model_name = model_name
        self.data_name = data_name
        self.privacy_name = privacy_name
        self.comment = '{}_{}'.format(model_name, data_name)
        self.data_subdir = '{}/{}'.format(model_name, data_name)
        # TensorBoard
        self.writer = SummaryWriter(comment="_" + self.comment + '_' + privacy_name)

    def log(self, autoencoder_loss, classifier_loss, adversary_loss, epoch, n_batch, num_batches, description):

        # var_class = torch.autograd.variable.Variable
        if isinstance(autoencoder_loss, torch.autograd.Variable):
            autoencoder_loss = autoencoder_loss.data.cpu().numpy()
        if isinstance(classifier_loss, torch.autograd.Variable):
            classifier_loss = classifier_loss.data.cpu().numpy()
        if isinstance(adversary_loss, torch.autograd.Variable):
            adversary_loss = adversary_loss.data.cpu().numpy()

        # step = Logger._step(epoch, n_batch, num_batches)
        step = epoch
        self.writer.add_scalar(
            '_{}/autoencoder_loss_{}'.format(self.comment, description), autoencoder_loss, step)
        self.writer.add_scalar(
            '_{}/classifier_loss_{}'.format(self.comment, description), classifier_loss, step)
        self.writer.add_scalar(
            '_{}/adversary_loss_{}'.format(self.comment, description), adversary_loss, step)

    def save_model(self, model, epoch):
        out_dir = './data/models/{}'.format(self.data_subdir)
        Logger._make_dir(out_dir)
        torch.save(model.state_dict(),
                   '{}/{}'.format(out_dir, self.data_name))

    def close(self):
        self.writer.close()

    @staticmethod
    def _step(epoch, n_batch, num_batches):
        return epoch * num_batches + n_batch

    @staticmethod
    def _make_dir(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def train_test_split2(X, y, S, test_size=0.3):
    split_size = int(X.shape[0] * test_size)
    X_test, y_test, s_test = X[0:split_size, :], y[0:split_size], S[0:split_size]
    X_train, y_train, s_train = X[split_size + 1:, :], y[split_size + 1:], S[split_size + 1:]
    print("Test size:", split_size)
    print("X shape:", X_train.shape, "y shape:", y_train.shape)
    return torch.from_numpy(X_train), torch.from_numpy(X_test), \
           torch.from_numpy(y_train), torch.from_numpy(y_test), \
           torch.from_numpy(s_train), torch.from_numpy(s_test)


class DatasetLoader(torch.utils.data.Dataset):
    """ Create traning data iterator """

    def __init__(self, feature_X, label_y, sentive_a):
        self.X = feature_X.double()
        self.y = label_y.double()
        self.A = sentive_a.double()
        if type(self.A) == np.ndarray:
            self.A = torch.from_numpy(self.A).double()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.X[idx, :], self.y[idx], self.A[idx]
