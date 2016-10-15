# Source : https://jeremykun.com/2013/10/28/optimism-in-the-face-of-uncertainty-the-ucb1-algorithm/
import math
from collections import defaultdict

class Bandit(object):
    
    def __init__(self):
        self.hist = []
        self.actions = set()
        self.nb_updates = 0

    def update(self, action, reward):
        raise NotImplementedError()
    
    def get_action_scores(self):
        raise NotImplementedError()

    def next(self):
        scores = self.get_action_scores()
        action, _ = max(scores.items(), key=lambda (action, score): score)
        return action 

class UCB(Bandit):
    
    def __init__(self):
        super(UCB, self).__init__()
        self.payoff_sums = defaultdict(lambda:0)
        self.num_plays = defaultdict(lambda:0)
        self.ucbs = defaultdict(lambda:0)

    def update(self, action, reward):
        self.actions.add(action)
        t = self.nb_updates
        self.num_plays[action] += 1
        self.payoff_sums[action] += reward
        self.nb_updates += 1

    def get_action_scores(self):
        assert self.nb_updates, 'You must provided at least one update iteration to get the next value'
        t = self.nb_updates
        po = self.payoff_sums
        nplays = self.num_plays
        actions = self.actions
        ucbs = {a: po[a] / nplays[a] + upperBound(t, nplays[a])  for a in actions}
        return ucbs

def upperBound(step, nplays):
    return math.sqrt(2 * math.log(step + 1) / nplays)

if __name__ == '__main__':
    import numpy as np
    ucb = UCB()
    simulate = {'a': lambda:np.random.normal(5), 'b': lambda:np.random.normal(1)}
    for _ in range(1):
        ucb.update('a', simulate['a']())
        ucb.update('b', simulate['b']())
    for _ in range(10):
        action = ucb.next()
        ucb.update(action, simulate[action]())
        action = ucb.next()
