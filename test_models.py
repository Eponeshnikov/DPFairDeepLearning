import os
import random
from threading import Thread

import run_training
comb_privacy = run_training.comb_privacy_eps
comb_architecture = run_training.comb_arch

# ====== Running parameters ======
parallel_threads = 3
repeats = 2
DATA_SET_NAME = 'Adult_1'
# ================================

thread_list = []
n_threads = 0
for j in range(repeats):
    for i in range(len(comb_privacy)):
        for k in range(len(comb_architecture)):
            seed = random.randint(0, 999999999999)
            n_threads += 1
            thread_list.append(Thread(target=os.system,
                                      args=('python run_training.py ' +
                                            DATA_SET_NAME + ' ' +
                                            str(i) + ' ' +
                                            str(k) + ' ' +
                                            str(seed),)))
            if n_threads % parallel_threads == 0 or \
                    (i + 1) * (k + 1) * (j + 1) == repeats * len(comb_privacy) * len(comb_architecture):
                for thread in thread_list:
                    print('thread start')
                    thread.start()
                for thread in thread_list:
                    thread.join()
                    print('thread end')
                print('=' * 40)
                thread_list = []
                n_threads = 0
