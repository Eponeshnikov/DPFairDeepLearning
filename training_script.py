import os
import numpy as np
import itertools
from dataset import Dataset
from utils import gen_exec_str
from dataclasses import make_dataclass
from threading import Thread
import time

# ====== Running parameters ======
parallel_threads = 1
repeats = 1
random_seed = False
no_cuda = False
# ================================
# ======= Model parameters =======
arch = ['DP']#, 'EOD']
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
advweight = [2]#[0.1, 0.2, 0.3, 0.5, 1, 2, 4]
xavier = [True]
# ================================
# ====== Dataset parameters ======
data_dir = 'dataset'
dataset = ['Adult']
batch = [1024*4]
sensattr = ['sex']
ages = [(71, 75)]
# ================================
# ====== Privacy parameters ======
privacy_in = [[]]
eps = [1]#, 3, 10, 30]
max_grad_norm = [1]
# ================================
# ====== Training parameters =====
epoch = [150]
adv_on_batch = [1]
eval_step_fair = [1]
grad_clip_ae = [10]
grad_clip_adv = [30]
grad_clip_class = [1]
optimizer_ae = ['Adam']
optimizer_adv = ['Adam']
optimizer_class = ['Adam']
lr_ae = [0.2]
lr_adv = [0.2]
lr_class = [0.2]
# ================================

all_exp = list(
    itertools.product(
        arch, edepth, ewidths, adepth, awidths, cdepth, cwidths, zdim, activ_ae, activ_adv, activ_class, e_activ_ae,
        e_activ_adv, e_activ_class, classweight, aeweight, advweight, xavier, dataset, batch, sensattr, ages,
        privacy_in,
        eps, max_grad_norm, epoch, adv_on_batch, eval_step_fair, grad_clip_ae, grad_clip_adv, grad_clip_class,
        optimizer_ae, optimizer_adv, optimizer_class, lr_ae, lr_adv, lr_class
    )
)

all_exp = [i for i in all_exp if 'sex' not in i] + [i for i in all_exp if 'sex' in i and ages[0] in i]
all_exp = [i for i in all_exp if ['autoencoder'] in i] + \
          [i for i in all_exp if ['autoencoder'] not in i and i[23] == eps[0]]

param_names = [
    'arch', 'edepth', 'ewidths', 'adepth', 'awidths', 'cdepth', 'cwidths', 'zdim', 'activ_ae', 'activ_adv',
    'activ_class', 'e_activ_ae', 'e_activ_adv', 'e_activ_class', 'classweight', 'aeweight', 'advweight', 'xavier',
    'dataset', 'batch', 'sensattr', 'ages', 'privacy_in', 'eps', 'max_grad_norm', 'epoch', 'adv_on_batch',
    'eval_step_fair', 'grad_clip_ae', 'grad_clip_adv', 'grad_clip_class',
    'optimizer_ae', 'optimizer_adv', 'optimizer_class', 'lr_ae', 'lr_adv', 'lr_class'
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
    thread_list.append(Thread(target=os.system, args=(gen_exec_str(p_l, param_names, seed, no_cuda),)))
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
