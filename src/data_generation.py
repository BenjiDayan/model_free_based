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

path = './osfstorage-archive/Experiment 1/twostep_data_study1/'
paths = [path + x for x in os.listdir(path) if x.endswith('.csv')]

fn = paths[0]

reward_probs_list = list(utils.subject_fn_to_reward_probs(fn))
# assert length of reward_probs_iterator is 200
# print(len(list(reward_probs_iterator)))
assert len(reward_probs_list) == 200

# is a list of series of form (r1, r2, r3, r4)
# stack them into one big Nx4 df
reward_probs_df = pd.concat([pd.DataFrame(x).T for x in reward_probs_list], ignore_index=True)



def unif_random_simplex_sample(dim: int):
    """sample a point from the simplex in dim dimensions"""
    # https://cs.stackexchange.com/questions/3227/uniform-sampling-from-a-simplex
    if dim == 1:
        return np.array([1])
    if dim < 1:
        raise ValueError("dim must be >= 1")
    z = np.random.uniform(size=dim-1)
    sorted_z = np.sort(z)
    return np.concatenate(([sorted_z[0]], np.diff(sorted_z), [1-sorted_z[-1]]))

def dist_buckets(num_buckets, num_balls):
    """num_buckets is the number of buckets to distribute num_balls into
    e.g. 3 buckets, 8 balls -> [3, 3, 2]
    e.g. 5 buckets, 21 balls -> [5, 4, 4, 4, 4]
    """
    outputs = []
    for _ in range(num_buckets):
        outputs.append(num_balls // num_buckets)
    missing = num_balls - sum(outputs)
    for i in range(missing):
        outputs[i] += 1
    return outputs


def unif_random_simplex_sample_with_0s(dim: int, total_samples=100, ratio=5, print_=False):
    """sample total_samples points from the simplex in dim dimensions.
    However we make sure to cover all the important points with dimensions zeroed out:
    
    sampled points will have 1, 2, ..., dim - 1, dim non-zeroes 
    dim choose 1, dim choose 2, ..., dim choose dim - 1 dimensions, dim choose dim ways
    and we want to sample quantities in the ratio of
    1 point, ratio points, ratio^2 points, ..., ratio^(dim-1) points
    1 (dim C 1 ) + ratio (dim C 2) + ... = (1 + ratio)^dim - 1 = Sigma
    So each bucket will actually get ~ (N / Sigma) * ratio^i points
    Except first (dim C 1) buckets will get 1 point each.

    e.g. num samples per bucket for dim=3, total_samples=30, ratio=3:
    Sigma=18.0
    (0,) 1
    (1,) 1
    (2,) 1
    (0, 1) 5
    (0, 2) 4
    (1, 2) 4
    (0, 1, 2) 14

    or if total_samples=7
    >>> list(unif_random_simplex_sample_with_0s(3, total_samples=7, ratio=3))
    Sigma=18.0
    (0,) 1
    (1,) 1
    (2,) 1
    (0, 1) 1
    (0, 2) 1
    (1, 2) 1
    (0, 1, 2) 1
    """

    # we ignore the first buckets, which is just 1 point each
    Sigma = (1/ratio) * ((1 + ratio)**dim - 1) - dim
    total_samples -= dim
    if print_:
        print(Sigma)

    total_balls = 0
    for num_non_zero in range(1, dim+1):
        combinations = list(itertools.combinations(range(dim), num_non_zero))
        num_buckets = len(combinations)
        if num_non_zero == 1:
            num_balls = num_buckets
        else:
            if num_non_zero == dim:
                num_balls = total_samples - total_balls
            else:
                one_per = math.comb(dim, num_non_zero)
                num_balls = int(np.floor(total_samples * (ratio**(num_non_zero-1)) * one_per / Sigma))
                if num_balls < one_per:
                    num_balls = min(one_per, total_samples - total_balls)
            total_balls += num_balls
        num_per_bucket = dist_buckets(num_buckets, num_balls)
        for non_zeroed_indices, num_samples in zip(combinations, num_per_bucket):
            if print_:
                print(non_zeroed_indices, num_samples)
            for _ in range(num_samples):
                output = np.zeros(dim)
                non_zeros = unif_random_simplex_sample(num_non_zero)
                for i, nzidx in enumerate(non_zeroed_indices):
                    output[nzidx] = non_zeros[i]
                yield output


### above as a dictionary
MODEL_KWARGS = {
    'beta_stage2': 8.0,
    'beta_mf0': 0.0,
    'beta_mf1': 0.0,
    'beta_mb': 10.0,
    'beta_stick': 0.0,
    'Q_MB_rare_prob': 0.3,
    'alpha': 0.15
}
        


outputs = unif_random_simplex_sample_with_0s(3, total_samples=8, ratio=4)

betas = np.array([1.0, 0.0, 0.0])
lam = 8.0
beta_mb, beta_mf0, beta_mf1 = betas * lam


from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from multiprocessing import Pool, Manager
def model_reward_single_trial(kwargs):
    # Scale betas by lambda, and remove lambda from kwargs
    lam = kwargs['lam']
    del(kwargs['lam'])
    kwargs['beta_mb'] = kwargs['beta_mb'] * lam
    kwargs['beta_mf0'] = kwargs['beta_mf0'] * lam
    kwargs['beta_mf1'] = kwargs['beta_mf1'] * lam
    model = models.Model(**kwargs)
    outs = model.perform_trials(
        reward_probs_list, save_Qs=False, save_probs=False, randomise=False)
    return outs

def model_reward(model_kwargs, n=35):
    # integrate given kwargs with default kwargs
    for k, v in MODEL_KWARGS.items():
        if k not in model_kwargs:
            model_kwargs[k] = v

    episodes_data = []
    for _ in range(n):
        episode_data = model_reward_single_trial(model_kwargs.copy())
        episode_data = episode_data.values.astype(np.uint8)
        episodes_data.append(episode_data)

    return episodes_data, model_kwargs

# TODO fix or delete
def model_reward_pooled(kwargs, n=35):
    model_kwargs2 = model_kwargs.copy()
    model_kwargs2.update(kwargs)
    with Pool(processes=8) as p:
        subject_mean_rewards = list(p.imap(model_reward_single_trial, (model_kwargs2 for _ in range(n))))

    return np.mean(subject_mean_rewards), np.std(subject_mean_rewards)/np.sqrt(n), model_kwargs2


def do_kwargs_star(args):
    return do_kwargs(*args)

import uuid
def write_df(df, lock=None, output_file=None, output_folder='./'):
    # if output_file is None, generate a uuid
    output_file = str(uuid.uuid4()) + '.csv' if output_file is None else output_file
    output_file = os.path.join(output_folder, output_file)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    def df_to_file(df, output_file):
        if not os.path.exists(output_file):
            df.to_csv(output_file, mode='w', header=True, index=False)
        else:
            df.to_csv(output_file, mode='a', header=False, index=False)

    # write file with lock if lock is not None
    if lock is not None:
        lock.acquire()
        try:
            df_to_file(df, output_file)
        finally:
            lock.release()
    else:
        df_to_file(df, output_file)


def do_kwargs(lock: Optional['lock'], model_kwargs, output_file: Optional[str], n=400, output_folder='./'):
    episodes_data, full_kwargs = model_reward(model_kwargs, n=n)    
    reward_rates = [episode_data[:, 3].mean() for episode_data in episodes_data]

    df_rows = pd.concat(\
        [pd.DataFrame([full_kwargs] * n),
         pd.Series(reward_rates),
         pd.Series(episodes_data)],
        axis=1)
    df_rows.columns = list(full_kwargs.keys()) + ['episode_reward_rate', 'episode_data']
    write_df(df_rows, lock=lock, output_file=output_file, output_folder=output_folder)

    return full_kwargs




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

betas_list = list(unif_random_simplex_sample_with_0s(3, 160, 5))
kwargs_list = []

if __name__ == '__main__':
    # output_file = 'data_generation_output_04_01_2024_episodic.csv'

    for alpha in alphas:
        for lam in lams:
            for my_betas in betas_list:
                # beta_mb, beta_mf0, beta_mf1 = lam * my_betas
                beta_mb, beta_mf0, beta_mf1 = my_betas
                kwargs = {
                    'beta_mb': beta_mb,
                    'beta_mf0': beta_mf0,
                    'beta_mf1': beta_mf1,
                    'lam': lam,
                    'alpha': alpha
                }
                # kwargs, output_file, n=35, output_folder='./'
                kwargs_list.append((kwargs, None, 35, 'data_generation_output_04_01_2024_episodic'))

    # with Manager() as manager:
    #     lock = manager.Lock()
    with Pool(processes=10) as p:
        results = list(tqdm(p.imap(do_kwargs_star, [(None, *args) for args in kwargs_list]), total=len(kwargs_list)))
                


##### TODO I think we actually care about episodes where reward rate was a particular smth, not just
# betas which on average give a particular reward rate.