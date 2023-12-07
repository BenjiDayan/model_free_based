import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def cbar(choice):
    """Gives the other choic: 0 -> 1, 1 -> 0"""
    return 1 - choice

### NB we initialise most Q(s,a) values with 0.5 as in rewards are given
# with some probability between 0.25 and 0.75, depending on the random walk

class Qclass:
    def __init__(self, beta=1.0, alpha=0.1):
        self.beta = beta
        self.alpha = alpha

    def get_val(self, key):
        pass

    def get_beta_scaled_val(self, key):
        return self.beta * self.get_val(key)

class Qstage2(Qclass):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Q[s][c]]
        self.Q = np.array([[0.5, 0.5], [0.5, 0.5]])

    def get_val(self, key):
        # key is a (s, c) tuple
        return self.Q[key]

    def update(self, state, choice2, r):
        # We learn a new Q value for the state and choice2. Tis the delta rule
        # Q_t+1(s, t) = (1-alpha) * Q_t(s, t) + alpha * r
        self.Q[(state, choice2)] = (1 - self.alpha) * self.Q[(state, choice2)] + self.alpha * r

class Q_MB(Qclass):
    def __init__(self, Qstage2, rare_prob=0.0, **kwargs):
        super().__init__(**kwargs)
        self.Qstage2 = Qstage2
        self.rare_prob = rare_prob

    def get_val(self, choice1):
        choice2 = cbar(choice1)
        # Gillan seems to use rare_prob=0.0, as if MB simplifies the situation and just thinks this way
        # DawDayan uses the "proper" rare_prob=0.3 as in the experiment.
        # Either way, the value of the choice is the max of the resultant Q values. Even if we don't always
        # choose the max of the two choices at the second stage.
        return (1-self.rare_prob) * max(self.Qstage2.Q[(choice1, 0)], self.Qstage2.Q[(choice1, 1)]) + \
                self.rare_prob * max(self.Qstage2.Q[(choice2, 0)], self.Qstage2.Q[(choice2, 1)])


class Q_MF0(Qclass):
    def __init__(self, Qstage2, **kwargs):
        super().__init__(**kwargs)

        # Q(choice1) where choice1 is in 0, 1
        self.Q = np.array([0.5, 0.5])

        self.Qstage2 = Qstage2

    def get_val(self, choice1):
        return self.Q[choice1]

    def update(self, choice1, stage2, choice2):
        # TD(0) update means we bootstrap from the value at the next stage. This is actually SARSA(0), as updates
        # to Q follow into changes in the policy.
        self.Q[choice1] = (1 - self.alpha) * self.Q[choice1] + self.alpha * self.Qstage2.Q[(stage2, choice2)]

    
class Q_MF1(Qclass):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Q(choice1)
        self.Q = np.array([0.5, 0.5])

    def get_val(self, choice1):
        return self.Q[choice1]

    def update(self, choice1, reward):
        # SARSA(1) means update as montecarlo for the whole episode, which is just the end reward.
        self.Q[choice1] = (1 - self.alpha) * self.Q[choice1] + self.alpha * reward


class Model:
    def __init__(self, alpha=0.1, beta_stage2=1.0, beta_mb=1.0, beta_mf0=1.0, beta_mf1=1.0, beta_stick=1.0):
        self.alpha = alpha

        self.beta_stick = beta_stick
        self.rare_prob = 0.3

        self.Qstage2 = Qstage2(beta=beta_stage2, alpha=alpha)
        self.Q_MB = Q_MB(self.Qstage2, beta=beta_mb)
        self.Q_MF0 = Q_MF0(self.Qstage2, beta=beta_mf0, alpha=alpha)
        self.Q_MF1 = Q_MF1(beta=beta_mf1, alpha=alpha)


        # State is which stage2 we end up in
    def get_stage2_action(self, state):
        """state is which stage2 we end up in"""
        # Q(st, c2,t)
        beta_scaled_qs = np.array([self.Qstage2.get_beta_scaled_val((state, choice2)) for choice2 in [0, 1]])
        probs = np.exp(beta_scaled_qs) / np.sum(np.exp(beta_scaled_qs))
        return np.random.choice([0, 1], p=probs)
        return probs
    
    def get_stage1_action(self, prev_choice1=None):
        """prev_choice1 is the choice we made in the previous trial - propensity to stick"""
        beta_scaled_qs = np.array(
            [
                [Q.get_beta_scaled_val(choice) for Q in [self.Q_MB, self.Q_MF0, self.Q_MF1]]
                  + [self.beta_stick if choice == prev_choice1 else 0]
                for choice in [0, 1]
            ])
        
        beta_scaled_qs = beta_scaled_qs.sum(axis=1)
        probs = np.exp(beta_scaled_qs) / np.sum(np.exp(beta_scaled_qs))
        return np.random.choice([0, 1], p=probs)
        return probs

    
    def perform_trial(self, drift1, drift2, drift3, drift4, prev_choice1=None):
        """for generating data."""
        choice1 = self.get_stage1_action(prev_choice1)
        stage2 = np.random.choice([choice1, cbar(choice1)], p=[1-self.rare_prob, self.rare_prob])
        choice2 = self.get_stage2_action(stage2)
        reward_prob = [[drift1, drift2], [drift3, drift4]][stage2][choice2]
        reward = np.random.choice([0, 1], p=[1-reward_prob, reward_prob])
        return choice1, stage2, choice2, reward
    
    def update(self, choice1, stage2, choice2, reward):
        self.Qstage2.update(stage2, choice2, reward)
        self.Q_MF0.update(choice1, stage2, choice2)
        self.Q_MF1.update(choice1, reward)

    def perform_trials(self, drifts_iterator, save_Qs=False):
        prev_choice1 = None
        outs = []
        for (drift1, drift2, drift3, drift4) in drifts_iterator:
            choice1, stage2, choice2, reward = self.perform_trial(drift1, drift2, drift3, drift4, prev_choice1=prev_choice1)
            self.update(choice1, stage2, choice2, reward)
            prev_choice1 = choice1
            if save_Qs:
                outs.append((choice1, stage2, choice2, reward, self.Qstage2.Q.copy(), np.array([self.Q_MB.get_val(0), self.Q_MB.get_val(1)]), self.Q_MF0.Q.copy(), self.Q_MF1.Q.copy()))
            else:
                outs.append((choice1, stage2, choice2, reward))

        return outs

        


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


def wrangle_df(df):
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


    df.loc[:, ['drift1', 'drift2', 'drift3', 'drift4']] = df.loc[:, ['drift1', 'drift2', 'drift3', 'drift4']].astype(float)

    # drop rows where reward is -1. These seem to be where the participant did not respond in time, i.e. RT=2502?
    df.reward = df.reward.astype(int)
    df = df[df.reward != -1]

    # reset index
    df.reset_index(drop=True, inplace=True)

    df.common_trans = df.common_trans.apply(lambda x: x == 'true')
    df.choice1_int = df.choice1_int.astype(int) - 1
    df.choice2_int = df.choice2_int.astype(int) - 1
    df.stage2 = df.stage2.astype(int) - 2

    return df


def qstage2_plot(qstage2_arr, df):
    """qstage2_arr is an array of shape (n_trials, 2, 2) where the last two dimensions are the Q values
    df has drift1, ..., drift4 columns"""

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
    """outs is a list of tuples of the form (choice1, stage2, choice2, reward, Qstage2, Q_MB, Q_MF0, Q_MF1)"""
    outs = pd.DataFrame(outs, columns=['choice1', 'stage2', 'choice2', 'reward', 'Qstage2', 'Q_MB', 'Q_MF0', 'Q_MF1'])
    # P(stay) for previous rewarded vs unrewarded trials, for common and rare transitions
    # T/F array
    stayed = outs['choice1'][1:].values == outs['choice1'][:-1].values
    # First trial is neither a stay nor a switch
    outs['stay'] = [None] + stayed.tolist()
    prev_rew = (outs['reward'][:-1].values == 1)
    outs['prev_rew'] = [None] + prev_rew.tolist()
    outs['common_trans'] = outs['stage2'] == outs['choice1']
    outs['prev_common_trans'] = [None] + outs['common_trans'][:-1].tolist()

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
    