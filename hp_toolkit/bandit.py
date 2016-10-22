# Source : https://jeremykun.com/2013/10/28/optimism-in-the-face-of-uncertainty-the-ucb1-algorithm/
import math
from functools import partial
from collections import defaultdict
from sklearn.utils import check_array, check_X_y
import numpy as np

from sklearn.preprocessing import FunctionTransformer
from scipy.stats import norm


class Bandit(object):
    
    def __init__(self, model):
        self.hist = []
        self.model = model
        self.actions = set()
        self.hist_actions = []
        self.hist_rewards = []
    
    def _update_actions(self, actions):
        self.actions |= set(actions)
    
    def _update_model(self, actions, rewards, partial_fit=False):
        self.hist_actions.extend(actions)
        self.hist_rewards.extend(rewards)
        if partial_fit:
            assert hasattr(model, 'partial_fit')
            self.model.partial_fit(actions, rewards)
        else:
            a = self.hist_actions
            r = self.hist_rewards
            self.model.fit(a, r)
    
    def update(self, action, reward):
        raise NotImplementedError()
    
    def get_action_scores(self, actions):
        raise NotImplementedError()

    def next(self, actions=None):
        actions, scores = self.get_action_scores(actions=actions)
        action, _ = max(zip(actions, scores), key=lambda (a,s):s)
        return action

def expected_improvement(model, X, hist_X=[], hist_y=[], xi=0.01):
    # source : scikit optimize
    y_opt = min(hist_y)
    mu, std = model.predict(X, return_std=True)
    values = np.zeros_like(mu)
    mask = std > 0
    improvement = y_opt - xi - mu[mask]
    exploit = improvement * norm.cdf(improvement / std[mask])
    explore = std[mask] * norm.pdf(improvement / std[mask])
    values[mask] = exploit + explore
    return values

class BayesianOptimization(Bandit):
    
    def __init__(self, model, criterion=expected_improvement):
        super(BayesianOptimization, self).__init__(model)
        self.criterion = criterion
 
    def update(self, actions, rewards):
        self._update_actions(actions)
        self._update_model(actions, rewards)

    def get_action_scores(self, actions):
        return self.criterion(self.model, actions, hist_X=self.hist_actions, hist_y=self.hist_rewards)

class UCB(Bandit):
    
    def __init__(self, model):
        super(UCB, self).__init__(model)
        self.payoff_sums = defaultdict(lambda:0)
        self.num_plays = defaultdict(lambda:0)

    def update(self, actions, rewards):
        self._update_actions(actions)
        self._update_model(actions, rewards)
        for a, r in zip(actions, rewards):
            self._update_one(a, r)

    def _update_one(self, action, reward):
        t = len(self.hist_rewards)
        self.num_plays[action] += 1
        self.payoff_sums[action] += reward

    def get_action_scores(self, actions=None):
        actions = actions if actions else self.actions
        return map(self._get_payoff, actions)

    def _get_payoff(self, action):
        po = self.payoff_sums
        nplays = self.num_plays
        a = action
        t = len(self.hist_rewards)
        if a in nplays:
            avg_reward = po[a] / nplays[a]
            ub = self._upperbound(t, nplays[a])
        else:
            avg_reward, std_reward = self.model.predict([a], return_std=True)
            avg_reward = avg_reward[0]
            ub = std_reward[0]
        return avg_reward + ub

    def _upperbound(self, step, nplays):
        return math.sqrt(2 * math.log(step + 1) / nplays)

class Simple(Bandit):
    
    def update(self, actions, rewards):
        self._update_actions(actions)
        self._update_model(actions, rewards)

    def get_action_scores(self, actions=None):
        actions = actions if actions else self.actions
        predict = self.model.predict
        actions = list(actions)
        rewards = predict(actions)
        return rewards


class Thompson(Bandit):
    
    def update(self, actions, rewards):
        self._update_actions(actions)
        self._update_model(actions, rewards)

    def get_action_scores(self, actions=None):
        actions = actions if actions else self.actions
        predict = self.model.sample_y
        actions = list(actions)
        rewards = predict(actions)
        return rewards

if __name__ == '__main__':
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.preprocessing import Imputer
    from helpers import DictVectorizer, Pipeline
    from frozendict import frozendict
    fd = frozendict
    gp = Pipeline(DictVectorizer(), Imputer(), GaussianProcessRegressor(normalize_y=True))
    bdt = Thompson(gp)
    x = [
        fd({'a': 1, 'b': 2}),
        fd({'a': 2, 'b': 3}),
        fd({'a': 5, 'b': 4})
    ]
    y = [3, 5, 9]
    bdt.update(x, y)
    actions = x + [fd({'a': 6, 'b': 4})]
    print(bdt.get_action_scores(actions=actions))
