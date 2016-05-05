from hp import instantiate_random
from sklearn.neighbors.kde import KernelDensity
import numpy as np
from collections import OrderedDict
from numpy.random import RandomState
from itertools import combinations
from collections import defaultdict
from heapq import heappush, heappop
from frozendict import frozendict

class Multinomial(object):

    def __init__(self, random_state=None):
        self.pr = OrderedDict()
        self.rng = RandomState(random_state)

    def fit(self, X):
        X = np.array(X)
        for name in sorted(set(X)):
            self.pr[name] = (X == name).mean()

    def sample(self, nb=1):
        return self.rng.choice(self.pr.keys(), p=self.pr.values())


class BaseSearch(object):

    def __init__(self, space, random_state=None):
        self.space = space
        self.rng = RandomState(random_state)
        self.hist = {"x": [], "y": []}

    def sample_next(self):
        raise NotImplementedError()

    def preprocess(self, x):
        return x

    def update(self, x, y):
        self.hist["x"].append(x)
        self.hist["y"].append(y)


class RandomSearch(BaseSearch):

    def sample_next(self):
        return instantiate_random(self.space, rng=self.rng)


class DensityFitSearch(BaseSearch):

    def __init__(self, space, epsilon=1, keep='all', random_state=None):
        """
        epsilon is the probability of exploring, instead of exploiting, it can be an iterator:
        def epsilon():
            t = 1
            while True:
                yield 1. / t

        keep is used in exploitation, if 'all' then use all the points on history
        to fit densities, else it uses only the best nth keep.
        """
        super(DensityFitSearch, self).__init__(space, random_state=random_state)
        self.keep = keep
        self.epsilon = epsilon
        self.sampler = {}
        self.xhist = []

    def feed(self, x):
        self.xhist.extend(x)

    def _fit(self, x):
        for name, param in self.space.items():

            v = [xval[name] for xval in x]
            if param.type == 'choice' or param.type == 'int':
                model = Multinomial()
                model.fit(v)
                self.sampler[name] = model
            else:
                density = KernelDensity(kernel='gaussian')
                v = np.array(v)
                v = v[:, None]
                density.fit(v)
                self.sampler[name] = density

    def sample_next(self):
        if len(self.hist["x"]):
            if self.keep == 'all':
                bestn = self.hist["x"]
            else:
                s = zip(*sorted(zip(self.hist["x"], self.hist["y"])))
                sort_x, sort_y = s
                bestn = list(sort_x)[0:self.keep]
        else:
            bestn = []
        try:
            epsilon = next(self.epsilon)
        except Exception:
            epsilon = self.epsilon

        try:
            keep = next(self.keep)
        except Exception:
            keep = self.keep

        if (self.rng.uniform() <= epsilon or 
            (self.keep != 'all' and len(bestn) < keep)):
            hp = instantiate_random(self.space, rng=self.rng)
            return hp
        else:
            self._fit(bestn)
            hp = instantiate_random(self.space, rng=self.rng)
            for name, s in self.sampler.items():
                hp[name] = s.sample()
                if self.space[name].type == 'int':
                    hp[name] = int(hp[name])
            return hp


class SubsetFeatureSearch(BaseSearch):

    def __init__(self, space, subset_size=1, a=0.5, b=0.5,
                 min_score=0, max_score=1,
                 sample_from_best=None, random_state=None):
        super(SubsetFeatureSearch, self).__init__(space, random_state=random_state)
        self.a = a
        self.b = b
        self.subset_size = subset_size
        self.xhist = []
        self.idx_to_name = {i: name for i, name in enumerate(space.keys())}
        self.combs = list(combinations(range(len(space)), subset_size))
        self.F = [0] * len(self.combs)
        self.S = [0] * len(self.combs)
        self.min_score = min_score
        self.max_score = max_score
        self.features = {}
        self.sample_from_best = sample_from_best

    def feed(self, x):
        self.xhist.extend(x)

    def sample_next(self):
        theta = []
        for i, comb in enumerate(self.combs):
            theta.append(self.rng.beta(self.S[i] + self.a, self.F[i] + self.b))
        arm = np.argmax(theta)
        features = self.combs[arm]
        hp = instantiate_random(self.space, rng=self.rng)
        hp_sub = {self.idx_to_name[idx]: hp[self.idx_to_name[idx]]
                  for idx in features}
        if self.sample_from_best is None:
            x = instantiate_random(self.space, rng=self.rng)
            self.feed([x])
            x = self.rng.choice(self.xhist)
        else:
            nb = self.sample_from_best
            if len(self.hist["x"]) >= nb:
                idxs = np.argsort(self.hist["y"])[::-1]
                idxs = idxs[0:nb]
                x = self.hist["x"][self.rng.choice(idxs)].copy()
            else:
                x = instantiate_random(self.space, rng=self.rng)
                self.feed([x])
        x.update(hp_sub)
        x["_arm"] = arm
        return x

    def preprocess(self, x):
        x_ = x.copy()
        del x_["_arm"]
        return x_

    def update(self, x, y):
        self.hist["x"].append(x)
        self.hist["y"].append(y)
        arm = x["_arm"]
        y_ = (y - self.min_score) / (self.max_score - self.min_score)
        self.S[arm] += 1 - y_
        self.F[arm] += y_


class Elite(BaseSearch):

    def __init__(self,
                 space,
                 subset_size=1, granularity=2, keep=1,
                 burnin=1,
                 random_state=None):
        super(Elite, self).__init__(space, random_state=random_state)
        self.subset_size = subset_size
        self.feature_subsets = list(combinations(space.keys(),
                                    self.subset_size))
        self.population = set()
        self.archive = defaultdict(lambda: defaultdict(list))
        self.granularity = granularity
        self.burnin = burnin
        self.keep = keep

    def sample_next(self):
        x1 = self._select()
        x2 = self._select()
        x = self._crossover(x1, x2)
        return x

    def _select(self):
        if len(self.population) <= self.burnin:
            x = instantiate_random(self.space, rng=self.rng)
        else:
            i = self.rng.choice(range(len(self.population)))
            x = list(self.population)[i]
        return x

    def _crossover(self, x1, x2):
        feats = self.space.keys()
        x1_feats = self.rng.choice(feats, len(feats)/2, replace=True)
        x2_feats = set(feats) - set(x1_feats)
        x = {}
        x.update({f: x1[f] for f in x1_feats})
        x.update({f: x2[f] for f in x2_feats})
        return x

    def update(self, x, y):
        self.hist["x"].append(x)
        self.hist["y"].append(y)

        for features in self.feature_subsets:
            bucket_str = ""
            for f in features:
                sp = self.space[f]
                if sp.type == 'real':
                    min_, max_ = sp.interval
                    val = (x[f] - min_) / (max_ - min_)
                    val = val * self.granularity
                    bucket = int(val)
                elif sp.type == 'int':
                    min_, max_ = sp.interval
                    val = x[f]
                    bucket = (val - min_) / self.granularity
                elif sp.type == 'choice':
                    val = sp.interval.index(x[f])
                    bucket = val
                bucket_str += str(bucket)
            bests = self.archive[features][bucket_str]
            heappush(bests, (y, frozendict(x)))
            if len(bests) > self.keep:
                _, val = heappop(bests)
                if val in self.population:
                    self.population.remove(val)
                _, curbest = max(bests)
                self.population.add(curbest)

    def bests(self):
        return self.population

if __name__ == "__main__":
    from hp_toolkit.hp import Param
    space = {
        "n": Param(initial=1, interval=[1, 20], type='int')
    }

    def evaluate(vals):
        return vals["n"] / 20.

    search = Elite(space, burnin=10, keep=1)

    for i in range(100):
        x = search.sample_next()
        y = evaluate(search.preprocess(x))
        search.update(x, y)
        print(x)
    print(search.bests())
