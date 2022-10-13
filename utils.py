import time
import torch
from collections import namedtuple
from dataclasses import make_dataclass
from clearml import Task, Logger


class CMLogger:
    def __init__(self, model_name, dataset_name):
        self.task = Task.init(project_name='AI Fairness',
                              task_name=f'{model_name}_{dataset_name}_{time.time()}')

        self.user_prop_dict = {"arch": model_name, "dataset": dataset_name}
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


def gen_namedtuple(args_dict, args_, name):
    named_tuple = make_dataclass(name, args_)
    specific_args_dict = {k: args_dict[k] for k in args_}
    return named_tuple(*specific_args_dict.values())


def gen_namedtuples(args_dict, name_and_args_dict):
    result = []
    for k in name_and_args_dict:
        result.append(gen_namedtuple(args_dict, name_and_args_dict[k], k))
    return result