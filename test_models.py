import os
from threading import Thread

import run_training
comb = run_training.comb_privacy_eps

n_threads = 0
parallel_threads = 1
thread_list = []
repeats = 1
DATA_SET_NAME = 'Adult_1_s'
for j in range(repeats):
    for i in range(len(comb)):
        n_threads += 1
        thread_list.append(Thread(target=os.system, args=('run_training.py ' + DATA_SET_NAME + ' ' + str(i),)))

        if n_threads % parallel_threads == 0 or i * (j + 1) == repeats * (len(comb) - 1):
            for thread in thread_list:
                print('thread start')
                thread.start()
            for thread in thread_list:
                thread.join()
                print('thread end')
            print('=' * 40)
            thread_list = []
            n_threads = 0
