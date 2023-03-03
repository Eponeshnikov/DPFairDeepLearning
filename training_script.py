import os
import numpy as np
import itertools
from dataset import Dataset
from utils import gen_exec_str
from dataclasses import make_dataclass
from threading import Thread
import time

# ====== Running parameters ======
parallel_threads = 2
repeats = 1
random_seed = True
no_cuda = False
check_acc = True
show_py_command = False
not_run = False
# ================================
# ======= Model parameters =======
arch = ['DP', 'EOD']
edepth = [2]
ewidths = [32]
adepth = [2]
awidths = [32]
cdepth = [2]
cwidths = [32]
zdim = [8]
activ_ae = ['leakyrelu']
activ_adv = ['leakyrelu']
activ_class = ['leakyrelu']
e_activ_ae = ['leakyrelu']
e_activ_adv = ['sigmoid']
e_activ_class = ['sigmoid']
classweight = [1]
aeweight = [0]
advweight = [1]
xavier = [True]
# ================================
# ====== Dataset parameters ======
data_dir = 'dataset'
dataset = ['Adult']
batch = ['max']
sensattr = ['sex']
ages = [(71, 75)]
# ================================
# ====== Privacy parameters ======
privacy_in = [['encoder_classifier'], ['encoder_classifier', 'adversary'], []]
eps = [1, 3, 10, 30]
max_grad_norm = [10]
# ================================
# ====== Training parameters =====
epoch = [10]
adv_on_batch = [1]
eval_step_fair = [10]
grad_clip_ae = [10]
grad_clip_adv = [10]
grad_clip_class = [10]
optimizer_enc_class = ['NAdam']
optimizer_adv = ['NAdam']
lr_enc_class = [0.11]
lr_adv = [0.11]
# ================================

all_exp = list(
    itertools.product(
        arch, edepth, ewidths, adepth, awidths, cdepth, cwidths, zdim, activ_ae, activ_adv, activ_class, e_activ_ae,
        e_activ_adv, e_activ_class, classweight, aeweight, advweight, xavier, dataset, batch, sensattr, ages,
        privacy_in,
        eps, max_grad_norm, epoch, adv_on_batch, eval_step_fair, grad_clip_ae, grad_clip_adv, grad_clip_class,
        optimizer_enc_class, optimizer_adv, lr_enc_class, lr_adv,
    )
)

private_parts = [i for i in privacy_in if len(i) > 0]
all_exp = [i for i in all_exp if 'sex' not in i] + [i for i in all_exp if 'sex' in i and ages[0] in i]
all_exp = [i for i in all_exp if any([j in i for j in private_parts])] + \
          [i for i in all_exp if not any([j in i for j in private_parts]) and i[23] == eps[0]]

param_names = [
    'arch', 'edepth', 'ewidths', 'adepth', 'awidths', 'cdepth', 'cwidths', 'zdim', 'activ_ae', 'activ_adv',
    'activ_class', 'e_activ_ae', 'e_activ_adv', 'e_activ_class', 'classweight', 'aeweight', 'advweight', 'xavier',
    'dataset', 'batch', 'sensattr', 'ages', 'privacy_in', 'eps', 'max_grad_norm', 'epoch', 'adv_on_batch',
    'eval_step_fair', 'grad_clip_ae', 'grad_clip_adv', 'grad_clip_class',
    'optimizer_enc_class', 'optimizer_adv', 'lr_enc_class', 'lr_adv'
]

if random_seed:
    seeds = np.random.randint(999999999, size=repeats * len(all_exp))
else:
    seeds = np.repeat(0, repeats * len(all_exp))

print(f'Total number of experiments: {len((all_exp * repeats))}')

# ======= Download datasets ======
for ds in dataset:
    dataset_args_ = make_dataclass('dataset_args', ['dataset', 'data_dir', 'only_download_data'])
    dataset_args = dataset_args_(ds, data_dir, True)
    d = Dataset(dataset_args)
    d.download_data()
# ================================

thread_list = []
n_threads = 0
start_time = time.time()
for i, (p_l, seed) in enumerate(zip((all_exp * repeats), seeds)):
    n_threads += 1
    command = gen_exec_str(p_l, param_names, seed, no_cuda, check_acc)
    if show_py_command:
        print(command)
    if not_run:
        continue
    thread_list.append(Thread(target=os.system, args=(command,)))
    if n_threads % parallel_threads == 0 or i == len(all_exp * repeats) - 1:
        for j, thread in enumerate(thread_list):
            print(f'thread starts ({i - (len(thread_list) - j) + 1})')
            thread.start()
        for j, thread in enumerate(thread_list):
            thread.join()
            print(f'thread ends ({i - (len(thread_list) - j) + 1})')
        print('=' * 40)
        thread_list = []
        n_threads = 0

print(f'Total time for job {round((time.time() - start_time) / 60, 2)} min')
