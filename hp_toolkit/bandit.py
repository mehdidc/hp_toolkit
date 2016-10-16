# Source : https://jeremykun.com/2013/10/28/optimism-in-the-face-of-uncertainty-the-ucb1-algorithm/
import math
from functools import partial
from collections import defaultdict
from sklearn.utils import check_array, check_X_y
import numpy as np

from sklearn.preprocessing import FunctionTransformer

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

class Thompson(Bandit):

    def __init__(self, model):
        super(Thompson, self).__init__()
        self.model = model
        self.hist_actions = []
        self.hist_rewards = []

    def update(self, action, reward, partial_fit=False):
        self.actions.add(action)
        self.hist_actions.append(action)
        self.hist_rewards.append(reward)
        if partial_fit:
            assert hasattr(model, 'partial_fit')
            self.model.partial_fit([action], [reward])
        else:
            a = self.hist_actions
            r = self.hist_rewards
            self.model.fit(a, r)

    def get_action_scores(self):
        predict = self.model.sample_y if hasattr(self.model, 'sample_y') else self.model.predict
        actions = list(self.actions)
        rewards = predict(actions)
        return {a: r for a, r in zip(actions, rewards)}

if __name__ == '__main__':
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.preprocessing import Imputer
    from helpers import DictVectorizer
    from frozendict import frozendict
    fd = frozendict
    gp = make_pipeline(DictVectorizer(), Imputer(), GaussianProcessRegressor())
    bdt = Thompson(gp)
    bdt.update(fd({'a': 1, 'b': 3}), 5)
    bdt.update(fd({'a': 2}), 5)
    bdt.update(fd({'a': 3, 'b': 4}), 8)
    print(bdt.get_action_scores())
