import collections
import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer

def flatten_dict(l):
    d = {}
    for k, v in l.items():
        if isinstance(v, collections.Mapping):
            d.update(flatten_dict(v))
        elif isinstance(v, list) or isinstance(v, tuple):
            for i, l in enumerate(v):
                if not isinstance(l, collections.Mapping):
                    d[k+'_{}'.format(i)] = l
                else:
                    for e in v:
                        d.update(flatten_dict(e))
        else:
            d[k] = v
    return d

DictFlattener = FunctionTransformer(flatten_dict)

class Pipeline(object):
    
    def __init__(self, *steps):
        self.steps = steps
    
    def fit_transform(self, X, y=None):
        self.fit(X, y=y)
        return self.transform(X, y=y)

    def fit(self, X, y=None):
        steps = self.steps
        for step in steps[0:-1]:
            X = step.fit_transform(X, y=y)
        return steps[-1].fit(X, y=y)

    def transform(self, X):
        steps = self.steps
        for step in steps[0:-1]:
            X = step.transform(X)
        return X
    
    def predict(self, X, *args, **kwargs):
        X = self.transform(X)
        y = self.steps[-1].predict(X, *args, **kwargs)
        return y

    def sample_y(self, X, *args, **kwargs):
        X = self.transform(X)
        y = self.steps[-1].sample_y(X, *args, **kwargs)
        return y


