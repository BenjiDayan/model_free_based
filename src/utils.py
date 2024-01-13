import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import models

# Peter: Start with generating not fitting.
# Then do logistic regression on thing.
# Get same gaussian walk - dig dig. Can generate own game.
# Actually would like the very precise random walk that they used.
# Not all random walks are equal.


def read_fn(fn):

    with open(fn) as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]
    i = min([i for i, x in enumerate(lines) if 'twostep_instruct_9' in x])
    lines = lines[i:]
    lines = [x.split(',') for x in lines]

    df = pd.DataFrame(lines[1:])
    return df


def wrangle_df(df, drop_minus_reward=True):
    # rename columns
    """A = trial_num
    B = drift 1 (probability of reward after second stage option 1)
    C = drift 2 (probability of reward after second stage option 2)
    D = drift 3 (probability of reward after second stage option 3)
    E = drift 4 (probability of reward after second stage option 4)
    F = stage 1 response (left/right)
    G = stage 1 selected stimulus (1/2 - note this is redundant with the response as the stage 1 options do not switch locations)
    H = stage 1 RT
    I = transition (common = TRUE; rare = FALSE)
    J = stage 2 response (left/right)
    K = stage 2 selected stimulus (1/2 - note this is redundant with response as the stage 2 options also do not switch locations)
    L =  stage 2 state (identity 2 or 3)
    M = stage 2 RT
    N = reward (1= yes; 0=no)
    O = redundant task variable, always set to 1"""

    columns = """A = trial_num
    B = drift1
    C = drift2
    D = drift3
    E = drift4
    F = choice1
    G = choice1_int
    H = stage1_RT
    I = common_trans
    J = choice2
    K = choice2_int
    L = stage2
    M = stage2_RT
    N = reward
    O = redundant"""
    columns = columns.split('\n')
    columns = [x.split(' = ')[1] for x in columns]

    df.columns = columns
    df.drop('redundant', axis=1, inplace=True)

    # reward probs to floats
    df.loc[:, ['drift1', 'drift2', 'drift3', 'drift4']] = \
        df.loc[:, ['drift1', 'drift2', 'drift3', 'drift4']].astype(float)

    # drop rows where reward is -1. These seem to be where the participant
    # did not respond in time, i.e. RT=2502?
    df.reward = df.reward.astype(int)
    if drop_minus_reward:
        df = df.loc[df.reward != -1].copy()
    df.reset_index(drop=True, inplace=True)

    # reset index
    df.reset_index(drop=True, inplace=True)

    df.common_trans = df.common_trans.apply(lambda x: x == 'true')
    # make all of these to be in 0, 1
    df.choice1_int = df.choice1_int.astype(int) - 1
    df.choice2_int = df.choice2_int.astype(int) - 1
    df.stage2 = df.stage2.astype(int) - 2

    return df


def qstage2_plot(qstage2_arr, df):
    """
    Plots stage2 reward probabilities (whether real or estimated) over time

    qstage2_arr is an array of shape (n_trials, 2, 2) where the last two dimensions are the Q values
    """

    # Do the previous four plots in one big figure, a 2x2 grid
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
    axes = axes.flatten()
    for i in range(4):
        _ = axes[i].plot(qstage2_arr[:, i//2, i%2], label='Qstage2_{}_{}'.format(i//2, i%2))
        _ = axes[i].plot(df.loc[:, 'drift{}'.format(i+1)], label='drift{}'.format(i+1))
        # df.loc[:, 'drift{}'.format(i+1)].plot(label='drift{}'.format(i+1))
        _ = axes[i].legend()
        
def qstage1_plot(df_outs):
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)

    q_mf0_arr = np.stack(df_outs.Q_MF0)
    axes[0][0].plot(q_mf0_arr[:, 0], label='q_mf0_0')
    axes[0][0].plot(q_mf0_arr[:, 1], label='q_mf0_1')

    q_mf1_arr = np.stack(df_outs.Q_MF1)
    axes[0][1].plot(q_mf1_arr[:, 0], label='q_mf1_0')
    axes[0][1].plot(q_mf1_arr[:, 1], label='q_mf1_1')

    q_mb_arr = np.stack(df_outs.Q_MB)
    axes[1][0].plot(q_mb_arr[:, 0], label='q_mb_0')
    axes[1][0].plot(q_mb_arr[:, 1], label='q_mb_1')

    axes[0][0].set_title('MF0: 0 blue vs 1 red')
    axes[0][1].set_title('MF1: 0 blue vs 1 red')
    axes[1][0].set_title('MB: 0 blue vs 1 red')


# drifts_iterator = (x[1] for x in df.loc[:, ['drift1', 'drift2', 'drift3', 'drift4']].iterrows())
# outs = model.perform_trials(drifts_iterator, save_Qs=True)

def outs_to_df(outs):
    """outs is a DF with columns (choice1, stage2, choice2, reward, Qstage2, Q_MB, Q_MF0, Q_MF1)"""
    # env_outs = outs['env']
    # outs = pd.DataFrame(outs, columns=['choice1', 'stage2', 'choice2', 'reward', 'Qstage2', 'Q_MB', 'Q_MF0', 'Q_MF1'])
    # P(stay) for previous rewarded vs unrewarded trials, for common and rare transitions
    # T/F array
    stayed = outs['choice1'][1:].values == outs['choice1'][:-1].values
    # All of these first we put as False just so that the dtype is still bool
    # We will only take 1: onwards anyway so first value doesn't matter
    outs['stay'] = [False] + stayed.tolist()
    prev_rew = (outs['reward'][:-1].values == 1)
    outs['prev_rew'] = [False] + prev_rew.tolist()
    outs['common_trans'] = outs['stage2'] == outs['choice1']
    outs['prev_common_trans'] = [False] + outs['common_trans'][:-1].tolist()

    return outs


def plot_barplot(df_outs):
    stay_prob_rew_common = df_outs.loc[df_outs.prev_rew & df_outs.prev_common_trans].stay
    stay_prob_rew_rare = df_outs.loc[df_outs.prev_rew & ~df_outs.prev_common_trans].stay
    stay_prob_unrew_common = df_outs.loc[~df_outs.prev_rew & ~df_outs.prev_common_trans].stay
    stay_prob_unrew_rare = df_outs.loc[~df_outs.prev_rew & ~df_outs.prev_common_trans].stay


    bar_width = 0.4

    # Plotting
    fig, ax = plt.subplots()
    bars1 = ax.bar([0.8, 2.8], [stay_prob_rew_common.mean(), stay_prob_unrew_common.mean()], bar_width, label='Common', capsize=5)
    bars2 = ax.bar([1.2, 3.2], [stay_prob_rew_rare.mean(), stay_prob_unrew_rare.mean()], bar_width, label='Rare', capsize=5)

    # ax.set_xlabel('Reward')
    ax.set_ylabel('P(stay)')
    # ax.set_title('Experiment 1')
    ax.set_xticks([1, 3])
    ax.set_xticklabels(['prev rewarded', 'prev unrewarded'])

    plt.legend()
    

def run_model_on_file(fn, model: models.Model):
    """fn is the filename of the csv file for a subject - contains
    the drifting reward probabilities for each trial. We will run the model as agent
    on this data (ignoring the actions and rewards of the actual subject)

    It turns out that all of Claire's subjects seem to get identically drifting reward probabilities,
    across 200 trials. So this is a bit unnecessary to do for each one but ah well
        - nice to have same for all subjects so randomness isn't a confounding factor
        - also partiuclar reward drifts might have been specially chosen as to have a nice structure,
          i.e. encouraging good learning and necessary of updating and changing strategy.
    
    Returns:
    subject_mean_stay_prob: list of 4 floats, mean P(stay) for each of the four conditions
    subject_mean_model_stay_prob: list of 4 floats, mean P(stay) for each of the four conditions
    df_outs: dataframe of the model outputs"""
    drifts_iterator = subject_fn_to_reward_probs(fn)
    outs = model.perform_trials(drifts_iterator, save_Qs=True, save_probs=True, randomise=False)

    df_outs_orig = outs_to_df(outs)
    df_outs = df_outs_orig.iloc[1:, :].copy()

    rew_common = df_outs.prev_rew & df_outs.prev_common_trans
    rew_rare = df_outs.prev_rew & ~df_outs.prev_common_trans
    unrew_common = ~df_outs.prev_rew & df_outs.prev_common_trans
    unrew_rare = ~df_outs.prev_rew & ~df_outs.prev_common_trans

    # actual stay or not, as well as the model's P(stay), for each of the four
    # trial types: rew com, rew rare, unrew com, unrew rare
    trial_stayed = [df_outs.loc[loccer].stay for loccer in [rew_common, rew_rare, unrew_common, unrew_rare]]
    df_outs['stay_prob'] = [x.probs1[x.choice1 if x.stay else models.cbar(x.choice1)] for _, x in \
                            df_outs.loc[:, ['probs1', 'choice1', 'stay']].iterrows()]
    

    # rew com, rew rare, unrew com, unrew rare
    model_mean_of_stay_probs = [np.mean(df_outs.loc[loccer].stay_prob) \
        for loccer in [rew_common, rew_rare, unrew_common, unrew_rare]]
    model_empirical_stay_prob = [np.mean(x) for x in trial_stayed]

    return model_empirical_stay_prob, model_mean_of_stay_probs, df_outs

def subject_fn_to_reward_probs(fn):
    df = read_fn(fn)
    # This actually removes trials where the subject did not respond in time
    df = wrangle_df(df, drop_minus_reward=False)

    # reward probs to floats
    df.loc[:, ['drift1', 'drift2', 'drift3', 'drift4']] = \
        df.loc[:, ['drift1', 'drift2', 'drift3', 'drift4']].astype(float)
    drifts_iterator = (x[1] for x in df.loc[:, ['drift1', 'drift2', 'drift3', 'drift4']].iterrows())
    return drifts_iterator

def run_model_simpler(model, reward_probs_iterator):
    """
    reward_probs_iterator: iterator of 4-tuples of floats, the reward probabilities for each
    of the four possible stage2 actions.

    Returns:
    Just list of model outputs (choice1, stage2, choice2, reward) x number trials

    We will use this to try to calibrate different models to have the same reward rate
    """
    # df with choice1, stage2, choice2, reward as columns, for 200 trials
    outs = model.perform_trials(
        reward_probs_iterator, save_Qs=False, save_probs=False, randomise=False)
    return outs




def plot_4_case_empirical_stay_probs(model_empirical_stay_probs):
    """
    model_empirical_stay_probs: List[List[float]]
        one 4 tuple per subject (model execution on reward probabilities), of the
        empirical stay probabilities for the four states
        rew common, rew rare, unrew common, unrew rare
    
    Plots a barplot of the mean empirical stay prob, and its standard error
    (std/sqrt(n_subjects)), which is an estimate of the error on this mean.
    """

    # "stay prob reward common", ...=
    sp_rc, sp_rr, sp_uc, sp_ur = zip(*model_empirical_stay_probs)

    # Plotting
    # n = len(sp_rew_common)

    bar_width = 0.4
    fig, ax = plt.subplots()
    bars1 = ax.bar([0.8, 1.8], [np.mean(sp_rc), np.mean(sp_uc)], bar_width, label='Common', capsize=5,
                yerr=[np.std(sp_rc)/np.sqrt(len(sp_rc)), np.std(sp_uc) / np.sqrt(len(sp_uc))])
    bars2 = ax.bar([1.2, 2.2], [np.mean(sp_rr), np.mean(sp_ur)], bar_width, label='Rare', capsize=5,
                yerr=[np.std(sp_rr) / np.sqrt(len(sp_rr)), np.std(sp_ur) / np.sqrt(len(sp_ur))])

    # ax.set_xlabel('Reward')
    ax.set_ylabel('P(stay)')
    # ax.set_title('Experiment 1')
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['prev rewarded', 'prev unrewarded'])

    # set y axes limits
    ax.set_ylim([0.3, 1.0])
    plt.legend()