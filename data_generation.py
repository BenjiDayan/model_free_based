# Here we will try to fit ~2000 models of different MB/MF ratios, with the aim
# of clamping the reward gained by each on the same 200 trials

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
model_kwargs = {
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

from multiprocessing import Pool
def model_reward_single_trial(kwargs):
    # Scale betas by lambda, and remove lambda from kwargs
    kwargs2 = kwargs.copy()
    lam = kwargs['lam']
    del(kwargs2['lam'])
    kwargs2['beta_mb'] = kwargs['beta_mb'] * lam
    kwargs2['beta_mf0'] = kwargs['beta_mf0'] * lam
    kwargs2['beta_mf1'] = kwargs['beta_mf1'] * lam
    model = models.Model(**kwargs2)
    outs = model.perform_trials(
        reward_probs_list, save_Qs=False, save_probs=False, randomise=False)
    return outs.reward.mean()

def model_reward(kwargs, n=35):
    # integrate given kwargs with default kwargs
    model_kwargs2 = model_kwargs.copy()
    model_kwargs2.update(kwargs)
    subject_mean_rewards = []
    # for _ in tqdm(range(n), total=n):
    for _ in range(n):
        subject_mean_rewards.append(model_reward_single_trial(model_kwargs2))

    return np.mean(subject_mean_rewards), np.std(subject_mean_rewards)/np.sqrt(n), model_kwargs2

def model_reward_pooled(kwargs, n=35):
    model_kwargs2 = model_kwargs.copy()
    model_kwargs2.update(kwargs)
    with Pool(processes=8) as p:
        subject_mean_rewards = list(p.imap(model_reward_single_trial, (model_kwargs2 for _ in range(n))))

    return np.mean(subject_mean_rewards), np.std(subject_mean_rewards)/np.sqrt(n), model_kwargs2


def do_kwargs_star(args):
    return do_kwargs(*args)

def do_kwargs(kwargs, output_file, n=400):
    rew_rate, rew_rate_stderr, full_kwargs = model_reward(kwargs, n=n)    
    outs.append((kwargs, rew_rate, rew_rate_stderr))
    # print confidence interval
    # print('; '.join([f'{k}: {v:.3f}' for k, v in kwargs.items()]))
    # print(f'CI: {rew_rate - 1.96 * rew_rate_stderr:.3f}, {rew_rate + 1.96 * rew_rate_stderr:.3f}')

    full_kwargs['rew_rate'] = rew_rate
    full_kwargs['rew_rate_stderr'] = rew_rate_stderr
    df_row = pd.DataFrame([full_kwargs], columns=full_kwargs.keys())
    # if is first time, write header
    if not os.path.exists(output_file):
        df_row.to_csv(output_file, mode='w', header=True, index=False)
    else:
        df_row.to_csv(output_file, mode='a', header=False, index=False)

    return rew_rate, rew_rate_stderr, full_kwargs

betas = np.array([1.0, 0.0, 0.0])
lam = 8.0
beta_mb, beta_mf0, beta_mf1 = betas * lam
outs = []

# alphas = [0.03, 0.05, 0.1, 0.15, 0.25, 0.4]
# alphas roughly 1.5x spaced apart, starting from 0.01
alphas = [0.01 * 1.6**i for i in range(9)]
# rounded to 4dp
alphas = [round(x, 4) for x in alphas]

# sigma(lambda * 0.3) vs sigma(lambda' * 0.3)
# if we want exp(lambda * 0.3) to be roughly 1.4x factors apart, we need
# lambda' = lambda + log(1.4) / 0.3 ~= lambda + 1.2
lams = [0.0, 0.5, 1.0] + [1.0 + 1.5 * i for i in range(1, 6)]

betas_list = list(unif_random_simplex_sample_with_0s(3, 30, 2))
kwargs_list = []
if __name__ == '__main__':
    output_file = 'data_generation_output.csv'

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
                kwargs_list.append((kwargs, output_file, 160))

    with Pool(processes=10) as p:
        results = list(tqdm(p.imap(do_kwargs_star, kwargs_list), total=len(kwargs_list)))
                



##### TODO I think we actually care about episodes where reward rate was a particular smth, not just
# betas which on average give a particular reward rate.