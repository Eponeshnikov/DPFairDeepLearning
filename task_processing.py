from tqdm import tqdm
import ast
import pickle
import os


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
        print(e)


def remote2local(tasks):
    local_tasks = [LocalTask(task) for task in tqdm(tasks)]
    return local_tasks


def save_tasks(tasks_list):
    for task in tqdm(tasks_list):
        save_dir = 'saved_tasks/'
        task_ = task
        if not isinstance(task, type(LocalTask(None))):
            task_ = LocalTask(task_)
        if 'archived' in task_.data_dict['system_tags']:
            save_dir += 'archived/'
        else:
            save_dir += 'current/'
        save_dir += task_.data_dict['id']
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for element in [(task_.params, 'params'), (task_.scalars, 'scalars'), (task_.data_dict, 'data_dict')]:
            el_name = element[1]
            with open(f"{save_dir}/{el_name}.pkl", 'wb') as f:
                pickle.dump(element[0], f)
