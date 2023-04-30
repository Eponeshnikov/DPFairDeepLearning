import time
import torch
import collections
from dataclasses import make_dataclass
from clearml import Task, Logger
import configparser
import json


class CMLogger:
    def __init__(self, model_name, dataset_name, credentials, exec_remote='', offline=False):
        if offline:
            Task.set_offline(offline_mode=True)
        api_server, web_server, files_server, access_key, secret_key = get_credentials(credentials[0], credentials[1])
        if api_server is not None:
            Task.set_credentials(
                api_host=api_server, web_host=web_server, files_host=files_server,
                key=access_key, secret=secret_key
             )
        else:
            print('Use default clearml.conf')
        Task.ignore_requirements('pywin32')
        self.task = Task.init(project_name='AI Fairness',
                              task_name=f'{model_name}_{dataset_name}_{time.time()}')
        self.task.set_archived(archive=False)
        self.user_prop_dict = {"arch": model_name, "dataset": dataset_name}
        self.task.set_parameters_as_dict(self.user_prop_dict)
        self.params_dictionary = {}
        self.task.connect(self.params_dictionary)
        self.logger = Logger.current_logger()
        if len(exec_remote) > 0:
            self.task.execute_remotely(queue_name=exec_remote)

    def log_metric(self, graph_name, metric_name, value, step):
        self.logger.report_scalar(graph_name, metric_name, value, step)

    def add_params(self, params):
        self.task.set_parameters_as_dict(params)


def get_credentials(dir_, file):
    if file != 'None':
        config = configparser.ConfigParser()
        config.read(f'{dir_}/{file}.conf')

        # Get the values of the different settings
        api_server = config.get('settings', 'api_server')
        web_server = config.get('settings', 'web_server')
        files_server = config.get('settings', 'files_server')

        # Parse the credentials JSON string
        credentials_json = config.get('settings', 'credentials')
        credentials = json.loads(credentials_json)
        access_key = credentials['access_key']
        secret_key = credentials['secret_key']
        return api_server, web_server, files_server, access_key, secret_key
    else:
        return None, None, None, None, None


def train_test_split(X, y, S, test_size=0.3):
    split_size = int(X.shape[0] * test_size)
    X_test, y_test, s_test = X[0:split_size, :], y[0:split_size], S[0:split_size]
    X_train, y_train, s_train = X[split_size + 1:, :], y[split_size + 1:], S[split_size + 1:]
    return X_train, X_test, y_train, y_test, s_train, s_test


def convert2torch(*arrs):
    result = [torch.from_numpy(x).float() for x in arrs]
    return result


def gen_dataclass(args_dict, args_, name):
    dataclass_ = make_dataclass(name, args_)
    specific_args_dict = {k: args_dict[k] for k in args_}
    return dataclass_(*specific_args_dict.values())


def gen_dataclasses(args_dict, name_and_args_dict):
    result = []
    for k in name_and_args_dict:
        result.append(gen_dataclass(args_dict, name_and_args_dict[k], k))
    return result


def gen_exec_str(param_list, param_names_, seed_, no_cuda_, check_acc_fair_, offline_mode_, python3=False):
    exec_str = 'python3 ' if python3 else 'python '
    exec_str += 'run_training.py'
    for p, n in zip(param_list, param_names_):
        if n == 'xavier':
            if p:
                exec_str += f' --{n}'
            else:
                pass
        elif n == 'privacy_in':
            for i in p:
                exec_str += f' --{n} {i}'
        elif n == 'ages':
            exec_str += f' --age_low {p[0]}'
            exec_str += f' --age_high {p[1]}'
        else:
            exec_str += f' --{n} {p}'
    exec_str += f' --seed {seed_}'
    if no_cuda_:
        exec_str += f' --no_cuda'
    if check_acc_fair_:
        exec_str += f' --check_acc_fair'
    if offline_mode_:
        exec_str += f' --offline_mode'
    return exec_str


def filter_by_cond(cond_, experiments):
    cond_ = collections.OrderedDict(sorted(cond_.items()))
    cond_zip = list(zip(*[cond_[i] for i in sorted(cond_.keys())]))
    tmp = []
    for condition in cond_zip:
        # get the index of the first element in the condition
        # compare the elements in "i" starting from the correct index
        filtered = [i for i in experiments if all([i[j] == condition[num] for num, j in enumerate(cond_)])]
        for f in filtered:
            if f not in tmp:
                tmp.append(f)
    return tmp


def filter_by_conds(conds_, experiments):
    result = []
    for cond_ in conds_:
        result = filter_by_cond(cond_, experiments)
        experiments = result.copy()
    return result
