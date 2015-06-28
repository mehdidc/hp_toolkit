import numpy as np
default_eval_functions = {"error": lambda model, X, y : (model.predict(X)!=y).mean()}
class Param(object):

    def __init__(self, initial, interval, type='real', scale='normal'):
        # types : real int choice
        # scale : normal log
        self.interval = interval
        self.initial = initial
        self.type = type
        self.scale = scale

def preprocessed_params(params, model_class):
    for k, v in params.items():
        if k in model_class.params:
            if model_class.params[k].type == 'int':
                params[k] = int(v)
    return params

def build_fn(model_class, eval_function,
             X_train, X_valid, y_train=None, y_valid=None,
             default_params=None):
    def fn(params):
        params = preprocessed_params(params, model_class)
        if default_params is not None:
            params.update(default_params)
        model = model_class(**params)
        model.fit(X_train, y_train)
        return eval_function(model, X_valid, y_valid)
    return fn

def find_best_hp(model_class,
                 minimize_fn,
                 X_train,
                 X_valid,
                 y_train=None,
                 y_valid=None,
                 allowed_params=None,
                 default_params=None,
                 eval_function=None,
                 max_evaluations=10):

    parameter_definition = dict()
    if eval_function is None:
        eval_function = lambda model, X_valid, y_valid: (model.predict(X_valid)!=y_valid).mean()
    fn = build_fn(model_class,
                  eval_function,
                  X_train, X_valid,
                  y_train, y_valid,
                  default_params=default_params)
    if allowed_params is None:
        params = model_class.params
    else:
        params = {p:model_class.params[p] for p in allowed_params}
    parameters, loss = minimize_fn((fn, max_evaluations, params))
    parameters = preprocessed_params(parameters, model_class)
    return parameters, loss

from hyperopt import hp, fmin, tpe, Trials
from hyperopt.mongoexp import MongoTrials
try:
    from pathos.multiprocessing import ProcessingPool as Pool
except ImportError:
    from multiprocessing import Pool

def parallelizer(pool_size=1):
    def wrap(minimizer):
        def minimizer_(params):
            p = Pool(pool_size)
            R = p.map(minimizer, [params] * pool_size)
            return min(R, key=lambda (params, loss): loss)
        return minimizer_
    return wrap

def minimize_fn_with_hyperopt(params):
    fn, max_evaluations, parameter_definition = params
    space = dict()
    for param_name, param_value in parameter_definition.items():
        if param_value.type == 'real':
            a, b = param_value.interval
            if param_value.scale == 'normal':
                space[param_name] = hp.uniform(param_name, a, b)
            elif param_value.scale.startswith('log'):
                if len(param_value.scale) > len('log'):
                    d = float(param_value.scale[len('log'):])
                else:
                    d = np.exp(1)
                space[param_name] = hp.loguniform(param_name, a *  np.log(d), b * np.log(d))
        elif param_value.type == 'int':
            a, b = param_value.interval
            if param_value.scale == 'normal':
                space[param_name] = hp.quniform(param_name, a, b, 1.)
            elif param_value.scale.startswith('log'):
                if len(param_value.scale) > len('log'):
                    d = int(param_value.scale[len('log'):])
                else:
                    d = np.exp(1)
                space[param_name] = hp.qloguniform(param_name, a * np.log(d), b * np.log(d))
        elif param_value.type == 'choice':
            space[param_name] = hp.choice(param_name, param_value.interval)
    trials = Trials()
    result = fmin(fn, space, algo=tpe.suggest,
                  max_evals=max_evaluations,
                  trials=trials)
    for param_name, param_value in parameter_definition.items():
        if param_value.type == 'choice':
            result[param_name] = param_value.interval[result[param_name]]
    return result, min(trials.losses())


def incorporate_params(inst, params):
    for k, v in inst.params.items():
        setattr(inst, k, params.get(k, v).initial)
    inst.__dict__.update(params)

class Model(object):

    def __init__(self, **params):
        self.state = 1234
        incorporate_params(self, params)

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.cross_validation import train_test_split

    class RandomForest(RandomForestClassifier):

        params = dict(
                max_depth=Param(initial=5, interval=[1, 10], type='int'),
                n_estimators=Param(initial=50, interval=[10, 300], type='int')
        )
    X, y = make_classification(n_samples=1000, n_features=20,
                               n_informative=2, n_redundant=2)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1)
    best_hp = find_best_hp(RandomForest,
                           parallelizer(20)(minimize_fn_with_hyperopt),
                           X_train, X_valid,
                           y_train, y_valid)
    print(best_hp)
