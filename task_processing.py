from numpy import ndarray
from tqdm import tqdm
import ast
import pickle
import os
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats


class LocalTask:
    def __init__(self, task_, remote=True):
        self.remote_task = task_
        self.params = dict()
        self.scalars = dict()
        self.data = None
        self.data_dict = None
        if remote:
            self.download_data()

    def get_parameters_as_dict(self):
        return self.params

    def get_reported_scalars(self):
        return self.scalars

    def download_data(self):
        try:
            self.scalars = self.remote_task.get_reported_scalars()
        except Exception as e:
            pass
        try:
            self.params = self.remote_task.get_parameters_as_dict()
        except Exception as e:
            pass
        try:
            self.data = self.remote_task.data
            self.data_dict = self.data.to_dict()
        except Exception as e:
            pass


class TaskListFilter:
    def __init__(self, task_list):
        self.task_list = task_list
        self.params = [task.get_parameters_as_dict() for task in task_list]
        self.scalars = [task.get_reported_scalars() for task in task_list]

    def filter_by_args(self, args_set, page='Args'):
        filtered_tasks = []
        for args in args_set:
            for param, task in zip(self.params, self.task_list):
                page = page if page in param.keys() else 'General'
                if all(key in param[page].keys() and
                       typeconv(param[page][key], value) == value for key, value in args.items()):
                    filtered_tasks.append(task)
        return filtered_tasks

    def extract_scalar_values(self, keys):
        values = []
        for scalar in self.scalars:
            nested_dict = scalar
            for key in keys:
                nested_dict = nested_dict[key]
            values.append(nested_dict)
        return values

    def extract_scalar_value(self, keys, id_=0):
        values = []
        nested_dict = self.scalars[id_]
        for key in keys:
            nested_dict = nested_dict[key]
        return nested_dict


class TaskListPipeline:
    def __init__(self, task_list):
        self.task_list = task_list
        self.operations = []

    def filter_by_args(self, *args, page='Args'):
        self.operations.append(lambda tasks: TaskListFilter(tasks).filter_by_args(list(args), page))
        return self

    def extract_scalar_values(self, keys):
        self.operations.append(lambda tasks: [TaskListFilter([task]).extract_scalar_value(keys) for task in tasks])
        return self

    def run(self):
        tasks = self.task_list
        for operation in self.operations:
            tasks = operation(tasks)
        return tasks


def typeconv(in_val, out_val):
    out_type = type(out_val)
    try:
        res_val = out_type(in_val)
        if out_type is list and len(in_val) > 0:
            res_val = ast.literal_eval(in_val)
        return res_val
    except Exception as e:
        try:
            res_val = out_type(float(in_val))
            return res_val
        except Exception as e:
            print(e)


def remote2local(tasks):
    local_tasks = [LocalTask(task) for task in tqdm(tasks)]
    return local_tasks


def save_tasks(tasks_list, add_imported_tag=False):
    for task in tqdm(tasks_list):
        save_dir = '../saved_tasks/'
        task_ = task
        if not isinstance(task, type(LocalTask(None))):
            task_ = LocalTask(task_)
        if 'archived' in task_.data_dict['system_tags']:
            save_dir += 'archived/'
        else:
            save_dir += 'current/'
        save_dir += task_.data_dict['name']
        save_dir += task_.data_dict['id']
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for element in [(task_.params, 'params'), (task_.scalars, 'scalars'), (task_.data_dict, 'data_dict')]:
            el_name = element[1]
            file = element[0]
            with open(f"{save_dir}/{el_name}.pkl", 'wb') as f:
                if el_name == 'data_dict' and add_imported_tag:
                    file['system_tags'] += ['imported']
                pickle.dump(file, f)


def read_local_pkl(path):
    with open(os.path.join(path, 'scalars.pkl'), 'rb') as f:
        scalars = pickle.load(f)
    with open(os.path.join(path, 'params.pkl'), 'rb') as f:
        params = pickle.load(f)
    with open(os.path.join(path, 'data_dict.pkl'), 'rb') as f:
        data_dict = pickle.load(f)
    return data_dict, params, scalars


def gen_bar_name(arch, privacy_in, awidth):
    name = f'LAFTR-{arch}|'
    if len(privacy_in) > 0:
        name += 'Privacy in '
        for pr in privacy_in:
            for el in pr.split('_'):
                name += f'{el}/'
        name = name[:len(name) - 1] + '|'
    else:
        name += 'No privacy|'
    if awidth == 32:
        name += 'Adversary = Classifier'
    elif awidth == 64:
        name += 'Adversary > Classsifier'
    return name


def map_sequence(x):
    res = np.interp(np.arange(0, x), (0, x), (-1, 1))
    return res, res[1] - res[0]


def add_no_privacy_bars(dataset: str,
                        task_list: List[str],
                        metrics: str,
                        privacy_ins: str,
                        archs: List[str],
                        awidths: List[int],
                        scalar='Model test'):
    plot_dict_no_privacy: Dict[str, List[float]] = dict()
    max_val = 0
    # Loop through each architecture and associated width
    for arch in archs:
        for awidth in awidths:
            # Create a pipeline without privacy
            pipeline_no_privacy = TaskListPipeline(task_list)

            # Filter the pipeline by specific arguments
            args = {'dataset': dataset,
                    'arch': arch,
                    'privacy_in': '',
                    'awidths': awidth}
            filtered_tasks_no_privacy_val = pipeline_no_privacy.filter_by_args(args).extract_scalar_values(
                [metrics, scalar, 'y', -1]).run()

            # Calculate the mean and standard deviation of filtered values
            mean_no_privacy = np.mean(filtered_tasks_no_privacy_val)
            std_no_privacy = np.std(filtered_tasks_no_privacy_val)
            max_val = np.max([max_val, mean_no_privacy + std_no_privacy])

            # Append to the plot_dict dictionary object
            plot_dict_no_privacy[gen_bar_name(arch, '', awidth)] = [mean_no_privacy, std_no_privacy,
                                                                    filtered_tasks_no_privacy_val]

    # Return the created dictionary object
    return plot_dict_no_privacy, max_val


# Define function to add unfair plot
def add_unfair_plot(dataset: str,
                    task_list: List[str],
                    metrics: str,
                    scalar='Unfair test') -> float:
    pipeline_unfair = TaskListPipeline(task_list)

    # Filter pipeline with dataset argument and take first value
    filtered_task_unfair = pipeline_unfair.filter_by_args({'dataset': dataset}).run()[0]

    # Extract scalar values for unfair test
    mean_unfair = TaskListPipeline([filtered_task_unfair]).extract_scalar_values(
        [metrics, scalar, 'y', -1]).run()

    # Return the mean result
    return mean_unfair


# Define function to add epsilon bars
def add_eps_bars(dataset: str,
                 task_list: List[str],
                 metrics: str,
                 privacy_ins: str,
                 archs: List[str],
                 awidths: List[int],
                 eps: List[float],
                 scalar='Model test'):
    plot_dict: Dict[str, List[float]] = dict()
    max_val = 0

    # Loop through each privacy-ins argument
    for privacy_in in privacy_ins:

        # Loop through each architecture and associated width
        for arch in archs:
            for awidth in awidths:
                # Create a pipeline of arguments
                pipeline = TaskListPipeline(task_list)

                # Filter the pipeline by specific arguments
                args = {'dataset': dataset,
                        'arch': arch,
                        'privacy_in': privacy_in,
                        'awidths': awidth}
                filtered_tasks = pipeline.filter_by_args(args).run()

                # Create a pipeline of the filtered tasks and filter by epsilon values
                eps_pipe = TaskListPipeline(filtered_tasks)
                eps_tasks_vals = eps_pipe.filter_by_args(*[{'eps': i} for i in eps]).extract_scalar_values(
                    [metrics, scalar, 'y', -1]).run()

                # Divide the list of epsilon values into sublists
                sub_lists = [eps_tasks_vals[
                             int(i * len(eps_tasks_vals) / len(eps)): int((i + 1) * len(eps_tasks_vals) / len(eps))] for
                             i in range(len(eps))]

                # Calculate means and standard deviations of the sublists and store in plot_dict
                means = [np.mean(i) for i in sub_lists]
                stds = [np.std(i) for i in sub_lists]
                max_val = np.max(np.hstack([max_val, [i + j for i, j in zip(means, stds)]]))
                plot_dict[gen_bar_name(arch, privacy_in, awidth)] = [means, stds, sub_lists]

    # Return the created dictionary object
    return plot_dict, max_val


# Define function to plot bars without privacy and unfair values
def plot_no_privacy_unfair(ax,
                           dataset: str,
                           task_list: List[str],
                           metrics: str,
                           privacy_ins: str,
                           archs: List[str],
                           awidths: List[int],
                           eps: List[float],
                           max_val: float,
                           scalar='Model test',
                           scalar_unfair='Unfair test'):
    # Set x values and spacing
    x = np.arange(len(eps)) * 2.5
    ddx = x[1] - x[0]

    # Add bars without privacy
    plot_dict_no_privacy, tmp_max_val = add_no_privacy_bars(dataset, task_list, metrics, privacy_ins,
                                                            archs, awidths, scalar=scalar)
    x_shift_no_pr, _ = map_sequence(len(plot_dict_no_privacy))
    [ax.bar(x[0] - ddx + x_shift_no_pr[i], k[1][0], yerr=k[1][1], label=k[0], width=_, alpha=0.5) for i, k in
     enumerate(plot_dict_no_privacy.items())]

    # Add unfair values
    mean_unfair = add_unfair_plot(dataset, task_list, metrics, scalar=scalar_unfair)
    ax.plot(np.array(
        [np.min(np.hstack([x[0] - ddx, x])) + np.min(x_shift_no_pr) - _ / 2,
         np.max(np.hstack([x[0] - ddx, x])) + np.max(x_shift_no_pr) + _ / 2]),
        [mean_unfair, mean_unfair],
        linestyle='dashed', color='black', label='Unfair|No privacy')

    # Set graph formatting
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    max_val = np.max([max_val, mean_unfair[0], tmp_max_val])
    ax.set_ylim(0, max_val * 1.05)
    ax.set_xlabel('ε')
    metrics = metrics if scalar == 'Model test' else f'Acc/Fair Δ{scalar}'
    ax.set_ylabel(metrics)
    ax.set_title(f'{dataset} dataset. Dependence of {metrics} on ε.')
    return plot_dict_no_privacy, mean_unfair


# Define function to plot eps bars
def plot_eps_bars(ax,
                  dataset: str,
                  task_list: List[str],
                  metrics: str,
                  privacy_ins: str,
                  archs: List[str],
                  awidths: List[int],
                  eps: List[float],
                  scalar='Model test'):
    # Add bars with privacy
    plot_dict, max_val = add_eps_bars(dataset, task_list, metrics, privacy_ins, archs, awidths, eps, scalar=scalar)

    # Set x values and spacing
    x = np.arange(len(eps)) * 2.5
    ddx = x[1] - x[0]

    # Set x axis tick labels
    ax.set_xticks(np.hstack([x[0] - ddx, x]))
    ax.set_xticklabels(['No privacy'] + eps)

    # Create bar graphs with labels
    x_shift, dx = map_sequence(len(plot_dict))
    [ax.bar(x + x_shift[i], k[1][0], yerr=k[1][1], label=k[0], width=dx, alpha=0.8) for i, k in
     enumerate(plot_dict.items())]

    # Set graph formatting
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_ylim(0, max_val * 1.05)
    ax.set_xlabel('ε')
    ax.set_ylabel(metrics)
    ax.set_title(f'{dataset} dataset. Dependence of {metrics} on ε.')
    return plot_dict, max_val


def convert_plot_dict(d, eps, mode='eps', repeat=10):
    dict_ = {'Name': [], 'Vals': []}
    if mode == 'eps':
        for k, v in d.items():
            for e_, e in zip(eps, v[2]):
                dict_['Name'].append(f"{k}|eps = {e_}")
                dict_['Vals'].append(e)
        return pd.DataFrame.from_dict(dict_)
    if mode == 'no_pr':
        for k, v in d.items():
            dict_['Name'].append(k)
            dict_['Vals'].append(v[2])
        return pd.DataFrame.from_dict(dict_)
    if mode == 'unfair':
        dict_['Name'].append('Unfair|No privacy')
        dict_['Vals'].append(np.repeat(d[0], repeat))
        return pd.DataFrame.from_dict(dict_)


def plot_ttest_matrix(res, dataset, metric, scalar='Model test', p_val=0.05):
    t = np.zeros((len(res), len(res)))
    for i in res.iterrows():
        for j in res.iterrows():
            vars_ = [np.var(i[1]['Vals']), np.var(j[1]['Vals'])]
            means = [np.mean(i[1]['Vals']), np.mean(j[1]['Vals'])]
            p_v = stats.ttest_ind(i[1]['Vals'], j[1]['Vals'], equal_var=not np.max(vars_) / np.min(vars_) > 4)[1]
            if p_v < p_val:
                if means[0] > means[1]:
                    t[i[0]][j[0]] = 1
                else:
                    t[i[0]][j[0]] = -1
            else:
                t[i[0]][j[0]] = 0
    f = plt.figure(figsize=(13, 13))
    ax = f.add_subplot(111)
    metric = metric if scalar == 'Model test' else f'Acc/Fair Δ{scalar}'
    ax.set_title(f'{dataset} dataset. {metric}.')
    a = ax.matshow(t)
    ax.xaxis.grid(True, color="black")
    ax.yaxis.grid(True, color="black")
    ax.set_xticks(np.arange(len(res['Name'])))
    ax.set_yticks(np.arange(len(res['Name'])))
    ax.set_xticklabels(res['Name'], rotation=90)
    ax.set_yticklabels(res['Name'])
