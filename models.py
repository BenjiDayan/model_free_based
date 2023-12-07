import numpy as np


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
                outs.append((choice1, stage2, choice2, reward, self.Qstage2.Q.copy(), self.Q_MF0.Q.copy(), self.Q_MF1.Q.copy()))
            else:
                outs.append((choice1, stage2, choice2, reward))

        return outs

        


# Peter: Start with generating not fitting.
# Then do logistic regression on thing.
# Get same gaussian walk - dig dig. Can generate own game.
# Actually would like the very precise random walk that they used.
# Not all random walks are equal.