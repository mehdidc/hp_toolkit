import collections
import numpy as np
import pandas as pd

class DictVectorizer(object):
    
    def __init__(self, colnames=None, missing_values='NaN'):
        self.colnames = set(colnames) if colnames else set()
        self.missing_values = np.nan if missing_values == 'NaN' else missing_values

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = map(flatten_dict, X)
        X = pd.DataFrame(X)
        self.colnames |= set(X.columns)
        for c in self.colnames - set(X.columns):
            X[c] = self.missing_values
        X = np.array(X)
        if len(X.shape) == 1:
            X = X[:, np.newaxis]
        return X

def flatten_dict(l):
    d = {}
    for k, v in l.items():
        if isinstance(v, collections.Mapping):
            d.update(flatten_dict(v))
        elif isinstance(v, list) or isinstance(v, tuple):
            for i, l in enumerate(v):
                d[k+'_{}'.format(i)] = l
        else:
            d[k] = v
    return d

