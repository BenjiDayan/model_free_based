# Here we will try to fit ~2000 models of different MB/MF ratios, with the aim
# of clamping the reward gained by each on the same 200 trials

from typing import Optional
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import importlib
import models
import utils
import numpy as np
import math

import itertools

path = '../osfstorage-archive/Experiment 1/twostep_data_study1/'
paths = [path + x for x in os.listdir(path) if x.endswith('.csv')]

fn = paths[0]

reward_probs_list = list(utils.subject_fn_to_reward_probs(fn))
# assert length of reward_probs_iterator is 200
# print(len(list(reward_probs_iterator)))
assert len(reward_probs_list) == 200

# is a list of series of form (r1, r2, r3, r4)
# stack them into one big Nx4 df
reward_probs_df = pd.concat([pd.DataFrame(x).T for x in reward_probs_list], ignore_index=True)



### above as a dictionary
MODEL_KWARGS = {
    'beta_stage2': 8.0,
    'beta_mf': 0.0,
    'beta_mb': 10.0,
    'beta_stick': 0.0,
    'Q_MB_rare_prob': 0.3,
    'alpha': 0.2
}
        


from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from multiprocessing import Pool, Manager
def model_reward_single_trial(reward_probs_list, kwargs):
    # Scale betas by lambda, and remove lambda from kwargs
    # lam = kwargs['lam']
    # del(kwargs['lam'])
    # kwargs['beta_mb'] = kwargs['beta_mb'] * lam
    # kwargs['beta_mf0'] = kwargs['beta_mf0'] * lam
    # kwargs['beta_mf1'] = kwargs['beta_mf1'] * lam
    model = models.Model_lam(**kwargs)
    outs = model.perform_trials(
        reward_probs_list, save_Qs=False, save_probs=False, randomise=False)
    return outs

def model_reward(reward_probs_list, model_kwargs, n=35):
    # integrate given kwargs with default kwargs
    for k, v in MODEL_KWARGS.items():
        if k not in model_kwargs:
            model_kwargs[k] = v

    episodes_data = []
    for _ in range(n):
        episode_data = model_reward_single_trial(reward_probs_list, model_kwargs.copy())
        episode_data = episode_data.values.astype(np.uint8)
        episodes_data.append(episode_data)

    return episodes_data, model_kwargs


def do_kwargs_star(args):
    return do_kwargs(*args)

import uuid
def write_df(df, lock=None, output_file=None, output_folder='./'):
    # if output_file is None, generate a uuid
    output_file = str(uuid.uuid4()) + '.csv' if output_file is None else output_file
    output_file = os.path.join(output_folder, output_file)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    def df_to_file(df, output_file):
        df.to_pickle(output_file)

    # write file with lock if lock is not None
    if lock is not None:
        lock.acquire()
        try:
            df_to_file(df, output_file)
        finally:
            lock.release()
    else:
        df_to_file(df, output_file)


def do_kwargs(lock: Optional['lock'], model_kwargs, output_file: Optional[str], n=400, output_folder='./', reward_probs_list=reward_probs_list):
    episodes_data, full_kwargs = model_reward(reward_probs_list, model_kwargs.copy(), n=n)    
    reward_rates = [episode_data[:, 3].mean() for episode_data in episodes_data]

    df_rows = pd.concat(\
        [pd.DataFrame([full_kwargs] * n),
         pd.Series(reward_rates),
         pd.Series(episodes_data)],
        axis=1)
    df_rows.columns = list(full_kwargs.keys()) + ['episode_reward_rate', 'episode_data']
    write_df(df_rows, lock=lock, output_file=output_file, output_folder=output_folder)

    return full_kwargs


def gen_gaussian_random_walk(lims=(0.25, 0.75), stdev=0.025, n=200):
    # reflects at outer limits
    start_point = np.random.uniform(*lims)
    out = [start_point]
    for _ in range(n-1):
        new_point = out[-1] + np.random.normal(0, stdev)
        if new_point > lims[1]:
            new_point = 2*lims[1] - new_point
        elif new_point < lims[0]:
            new_point = 2*lims[0] - new_point
        out.append(new_point)

    return np.array(out)

def gen_fake_reward_probs():
    reward_probs = np.array([gen_gaussian_random_walk() for _ in range(4)])
    reward_probs = reward_probs.transpose(1, 0)  # 200, 4
    return reward_probs


betas = np.array([1.0, 0.0, 0.0])
lam = 8.0
beta_mb, beta_mf0, beta_mf1 = betas * lam
outs = []

# alphas = [0.03, 0.05, 0.1, 0.15, 0.25, 0.4]
# alphas roughly 1.5x spaced apart, starting from 0.01
alphas = [0.01 * 1.2**i for i in range(20)]
# rounded to 4dp
alphas = [round(x, 4) for x in alphas]

# sigma(lambda * 0.3) vs sigma(lambda' * 0.3)
# if we want exp(lambda * 0.3) to be roughly 1.4x factors apart, we need
# lambda' = lambda + log(1.4) / 0.3 ~= lambda + 1.2
lams = [0.0, 0.1, 0.3, 0.6] + [0.6 + 0.5 * i for i in range(1, 20)]


if __name__ == '__main__':
    output_folder = 'kastner_generation_13_01_2024_mb'
    kwargs_list = []

    ##### many new reward_probs_lists
    kwargs = {
        'beta_mb': 8.0,
        'beta_mf': 0.0,
    }
    for _ in range(500):
        rps = gen_fake_reward_probs()
        o_file_uuid = str(uuid.uuid4())
        output_file  = o_file_uuid + '.csv'
        os.makedirs(output_folder, exist_ok=True)
        np.save(os.path.join(output_folder, o_file_uuid + '.npy'), rps)
        kwargs_list.append((kwargs, output_file, 100, output_folder, rps))

    with Pool(processes=10) as p:
        results = list(tqdm(p.imap(do_kwargs_star, [(None, *args) for args in kwargs_list]), total=len(kwargs_list)))
    exit()


    ##### b)
    # 5001 action sequences for MB/MF weighting being
    # 0, 1/5000, 2/5000, ..., 5000/5000

    for beta_mb in np.linspace(0, 8, 5001):
        kwargs = {
            'beta_mb': beta_mb,
            'beta_mf': 8.0 - beta_mb,
        }
        kwargs_list.append((kwargs, None, 1, output_folder))


    ###### a)
    # 500 action sequences each of pure MB and pure MF
    kwargs = {
        'beta_mb': 0.0,
        'beta_mf': 8.0,
    }
    kwargs_list.append((kwargs, None, 500, output_folder))

    kwargs = {
        'beta_mb': 8.0,
        'beta_mf': 0.0,
    }
    kwargs_list.append((kwargs, None, 500, output_folder))

    ##### c) 
    # 20100 action sequences for 201 values of MB/MF weighting
    # crossed with 100 values of learning rate alpha 0.01 ... 0.8
    
    for beta_mb in np.linspace(0, 8, 201):
        for alpha_exp in np.linspace(np.log(0.01), np.log(0.8), 100):
            alpha = np.exp(alpha_exp)
            kwargs = {
                'beta_mb': beta_mb,
                'beta_mf': 8.0 - beta_mb,
                'alpha': alpha
            }
            kwargs_list.append((kwargs, None, 1, output_folder))


    with Pool(processes=10) as p:
        results = list(tqdm(p.imap(do_kwargs_star, [(None, *args) for args in kwargs_list]), total=len(kwargs_list)))
                


##### TODO I think we actually care about episodes where reward rate was a particular smth, not just
# betas which on average give a particular reward rate.
