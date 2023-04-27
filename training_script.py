import os
import numpy as np
import itertools
from dataset import Dataset
from utils import gen_exec_str, filter_by_conds
from dataclasses import make_dataclass
from threading import Thread
import time

# ====== Running parameters ======
parallel_threads = 4
repeats = 2
random_seed = True
no_cuda = False
check_acc_fair = True
check_acc_fair_attempts = [20]  # 40
acc_tresh = [0.5]  # 44
dp_atol = [0.02]  # 45
eod_atol = [0.02]  # 45
show_py_command = False
not_run = False
continue_from = 0
test_mode = 0  # |~(16, 34)~|
offline_mode = False  # !!!Not works with check_acc_fair if condition pass!!!
config_dir = ['configs']  # 42
server = ['local_server']  # 43
# ================================
# ======= Model parameters =======
arch = ['DP', 'EOD']  # 0
edepth = [2]  # 1
ewidths = [32]  # 2
adepth = [2, 4]  # 3
awidths = [32, 64]  # 4
cdepth = [2]  # 5
cwidths = [32]  # 6
zdim = [8]  # 7
activ_ae = ['leakyrelu']  # 8
activ_adv = ['leakyrelu']  # 9
activ_class = ['leakyrelu']  # 10
e_activ_ae = ['leakyrelu']  # 11
e_activ_adv = ['sigmoid']  # 12
e_activ_class = ['sigmoid']  # 13
classweight = [1]  # 14
aeweight = [0]  # 15
advweight = [1]  # 16
xavier = [True]  # 17
# ================================
# ====== Dataset parameters ======
data_dir = ['dataset']  # 41
dataset = ['German']  # 18
batch = ['max']  # 19
sensattr = ['sex']  # 20
ages = [(71, 75)]  # 21
# ================================
# ====== Privacy parameters ======
privacy_in = [['encoder_classifier'], ['encoder_classifier', 'adversary'], []]  # 22
eps = [1, 3, 10, 30]  # 23
max_grad_norm = [10]  # 24
# ================================
# ====== Training parameters =====
epoch = [500]  # 25
adv_on_batch = [1]  # 26
eval_step_fair = [10]  # 27
grad_clip_ae = [10]  # 28
grad_clip_adv = [10]  # 29
grad_clip_class = [10]  # 30
optimizer_enc_class = ['NAdam']  # 31
optimizer_adv = ['NAdam']  # 32
lr_enc_class = [0.14, 0.12]  # 33
lr_adv = [0.14, 0.12]  # 34
enc_class_sch = ['PolynomialLR']  # 35
adv_sch = ['PolynomialLR']  # 36
enc_class_sch_pow = [1]  # 37
adv_sch_pow = [1]  # 38
eval_model = ['LR']  # 39
# ========== Link params =========
conds = [{4: awidths, 3: adepth, 33: lr_enc_class, 34: lr_adv},
         {18: dataset, 25: epoch}]
# ================================

all_exp = list(
    itertools.product(
        arch, edepth, ewidths, adepth, awidths, cdepth, cwidths, zdim, activ_ae, activ_adv, activ_class, e_activ_ae,
        e_activ_adv, e_activ_class, classweight, aeweight, advweight, xavier, dataset, batch, sensattr, ages,
        privacy_in, eps, max_grad_norm, epoch, adv_on_batch, eval_step_fair, grad_clip_ae, grad_clip_adv,
        grad_clip_class, optimizer_enc_class, optimizer_adv, lr_enc_class, lr_adv, enc_class_sch, adv_sch,
        enc_class_sch_pow, adv_sch_pow, eval_model, check_acc_fair_attempts, data_dir, config_dir, server,
        acc_tresh, dp_atol, eod_atol
    )
)

# Delete unnecessary args (sensattr)
all_exp = [i for i in all_exp if 'sex' not in i] + [i for i in all_exp if 'sex' in i and ages[0] in i]
# Delete unnecessary args (eps) if no privacy
private_parts = [i for i in privacy_in if len(i) > 0]
all_exp = [i for i in all_exp if any([j in i for j in private_parts])] + \
          [i for i in all_exp if not any([j in i for j in private_parts]) and i[23] == eps[0]]

# Sort by dataset
all_exp = [[i for i in all_exp if i[18] == j] for j in dataset]
all_exp = [item for sublist in all_exp for item in sublist]

if len(conds) > 0:
    all_exp = filter_by_conds(conds, all_exp)

if test_mode > 0:
    all_exp = [all_exp[test_mode]]

param_names = [
    'arch', 'edepth', 'ewidths', 'adepth', 'awidths', 'cdepth', 'cwidths', 'zdim', 'activ_ae', 'activ_adv',
    'activ_class', 'e_activ_ae', 'e_activ_adv', 'e_activ_class', 'classweight', 'aeweight', 'advweight', 'xavier',
    'dataset', 'batch', 'sensattr', 'ages', 'privacy_in', 'eps', 'max_grad_norm', 'epoch', 'adv_on_batch',
    'eval_step_fair', 'grad_clip_ae', 'grad_clip_adv', 'grad_clip_class', 'optimizer_enc_class', 'optimizer_adv',
    'lr_enc_class', 'lr_adv', 'enc_class_sch', 'adv_sch', 'enc_class_sch_pow', 'adv_sch_pow', 'eval_model',
    'check_acc_fair_attempts', 'data_dir', 'config_dir', 'server', 'acc_tresh', 'dp_atol', 'eod_atol'
]

if random_seed:
    seeds = np.random.randint(999999999, size=repeats * len(all_exp))
else:
    seeds = np.repeat(0, repeats * len(all_exp))

print(f'Total number of experiments: {len((all_exp * repeats)[continue_from:])}')

# ======= Download datasets ======
for ds in dataset:
    for data_dir_ in data_dir:
        dataset_args_ = make_dataclass('dataset_args', ['dataset', 'data_dir', 'only_download_data'])
        dataset_args = dataset_args_(ds, data_dir_, True)
        d = Dataset(dataset_args)
        d.download_data()
# ================================

thread_list = []
n_threads = 0
start_time = time.time()
for i, (p_l, seed) in enumerate(zip((all_exp * repeats)[continue_from:], seeds[continue_from:])):
    n_threads += 1
    command = gen_exec_str(p_l, param_names, seed, no_cuda, check_acc_fair, offline_mode)
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
