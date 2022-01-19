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

    def __init__(self, model_name, data_name):
        self.model_name = model_name
        self.data_name = data_name

        self.comment = '{}_{}'.format(model_name, data_name)
        self.data_subdir = '{}/{}'.format(model_name, data_name)

        # TensorBoard
        self.writer = SummaryWriter(comment=self.comment)

    def log(self, ae_error, class_error, adv_error, epoch, n_batch, num_batches, description='train'):

        # var_class = torch.autograd.variable.Variable
        if isinstance(ae_error, torch.autograd.Variable):
            ae_error = ae_error.data.cpu().numpy()
        if isinstance(class_error, torch.autograd.Variable):
            class_error = class_error.data.cpu().numpy()
        if isinstance(adv_error, torch.autograd.Variable):
            adv_error = adv_error.data.cpu().numpy()

        step = Logger._step(epoch, n_batch, num_batches)
        self.writer.add_scalar(
            '_{}/ae_error_{}'.format(self.comment, description), ae_error, step)
        self.writer.add_scalar(
            '_{}/class_error_{}'.format(self.comment, description), class_error, step)
        self.writer.add_scalar(
            '_{}/adv_error_{}'.format(self.comment, description), adv_error, step)

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
  X_train, y_train, s_train  = X[split_size+1:, :], y[split_size+1:], S[split_size+1:]
  print(split_size)
  print(X_train.shape, y_train.shape)
  return torch.from_numpy(X_train), torch.from_numpy(X_test), torch.from_numpy(y_train), torch.from_numpy(y_test), torch.from_numpy(s_train), torch.from_numpy(s_test)


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