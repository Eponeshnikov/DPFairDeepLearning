import time
import numpy as np
import torch
from clearml import Task, Logger

'''
    TensorBoard Data will be stored in './runs' path
'''


class CMLogger:
    def __init__(self, model_name, dataset_name):
        self.task = Task.init(project_name='AI Fairness',
                              task_name=model_name + '_' + dataset_name + str(time.time()))

        self.user_prop_dict = {"Model name": model_name, "Dataset": dataset_name}
        self.task.set_parameters_as_dict(self.user_prop_dict)
        self.params_dictionary = {}
        self.task.connect(self.params_dictionary)
        self.logger = Logger.current_logger()

    def log_metric(self, graph_name, metric_name, value, step):
        self.logger.report_scalar(graph_name, metric_name, value, step)

    def log_plt(self, title, series, epoch, fig):
        self.logger.report_matplotlib_figure(title, series, iteration=epoch, figure=fig)

    def add_params(self, params):
        self.task.set_parameters_as_dict(params)


def train_test_split(X, y, S, test_size=0.3):
    split_size = int(X.shape[0] * test_size)
    X_test, y_test, s_test = X[0:split_size, :], y[0:split_size], S[0:split_size]
    X_train, y_train, s_train = X[split_size + 1:, :], y[split_size + 1:], S[split_size + 1:]
    return X_train, X_test, y_train, y_test, s_train, s_test


def convert2torch(*arrs):
    result = [torch.from_numpy(x).float() for x in arrs]
    return result


class DatasetLoader(torch.utils.data.Dataset):
    """ Create traning data iterator """

    def __init__(self, feature_X, label_y, sentive_a):
        self.X = feature_X.float()
        self.y = label_y.float()
        self.A = sentive_a.float()
        if type(self.A) == np.ndarray:
            self.A = torch.from_numpy(self.A).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.X[idx, :], self.y[idx], self.A[idx]
