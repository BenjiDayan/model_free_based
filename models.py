import numpy as np

class Qstage2:
    def __init__(self, beta=1.0, alpha=0.1):
        self.alpha = alpha
        self.beta = beta

        # Q(st, c2,t)
        self.Q = {(a, b): 0 for a in [0, 1] for b in [0, 1]} 

    def update(self, state, choice2, r):
        # Q(st, c2,t)
        self.Q[(state, choice2)] = (1 - self.alpha) * self.Q[(state, choice2)] + self.alpha * r

    # State is which stage2 we end up in
    def get_action(self, state):
        # Q(st, c2,t)
        q = np.array([self.Q[(state, choice2)] for choice2 in [0, 1]])
        probs = np.exp(self.beta * q) / np.sum(np.exp(self.beta * q))
        return np.random.choice([0, 1], p=probs)

class Q_MB:
    def __init__(self, Qstage2, beta=1.0):
        self.Qstage2 = Qstage2
        self.beta = beta

    @property
    def Q(self, choice1):
        return max(self.Qstage2.Q[(choice1, 0)], self.Qstage2.Q[(choice1, 1)])


class Q_MF0:
    def __init__(self, beta=1.0, alpha=0.1):
        self.alpha = alpha
        self.beta = beta

        # Q(choice1)
        self.Q = {0: 0, 1: 0}

    def update(self, choice1, stage2, choice2):
        self.Q[choice1] = (1 - self.alpha) * self.Q[choice1] + self.alpha * self.Qstage2.Q[(stage2, choice2)]

    
class Q_MF1:
    def __init__(self, beta=1.0, alpha=0.1):
        self.alpha = alpha
        self.beta = beta

        # Q(choice1)
        self.Q = {0: 0, 1: 0}

    def update(self, choice1, reward):
        self.Q[choice1] = (1 - self.alpha) * self.Q[choice1] + self.alpha * reward


class Model:
    def __init__(self):  # TODO