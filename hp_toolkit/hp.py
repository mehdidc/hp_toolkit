import numpy as np

from hyperopt import hp, fmin, tpe, Trials
from hyperopt.pyll.stochastic import sample
import logging

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


def classification_error(model, X, y):
    return (model.predict(X) != y).mean()


default_eval_functions = {
        "error": classification_error
}


class Param(object):

    def __init__(self, initial, interval, type='real', scale='normal'):
        # types : real int choice
        # scale : normal log
        self.interval = interval
        self.initial = initial
        self.type = type
        self.scale = scale


def make_constant_param(value):
    return Param(initial=value, interval=[value],
                 type='choice')


def preprocessed_params(params_inst, params):
    for k, v in params_inst.items():
        if k in params:
            if params[k].type == 'int':
                params_inst[k] = int(v)
    return params_inst


def preprocessed_params_from_model_class(params, model_class):
    return preprocessed_params(params, model_class.params)


def build_fn(model_class, eval_function,
             X_train, X_valid, y_train=None, y_valid=None,
             default_params=None,
             model_init_func=None):
    def fn(params):
        params = preprocessed_params_from_model_class(params, model_class)
        if default_params is not None:
            params.update(default_params)

        logging.info("Trying : {0}".format(str(params)))
        model = model_class(**params)
        if model_init_func is not None:
            opt = dict(
                    X_train=X_train,
                    X_valid=X_valid,
                    y_train=y_train,
                    y_valid=y_valid
            )
            model = model_init_func(model, **opt)
        model.fit(X_train, y_train)
        loss = eval_function(model, X_valid, y_valid)
        if np.isnan(loss):
            logging.warning("NaN in the loss with  : {0}".format(str(params)))
            return dict(status='fail', hp=params)
        else:
            logging.info("Evaluation of {0} : {1}".format(str(params), loss))
            return dict(loss=loss,
                        status="ok",
                        hp=params)
    return fn


def find_all_hp(model_class,
                minimize_fn,
                X_train,
                X_valid,
                y_train=None,
                y_valid=None,
                allowed_params=None,
                not_allowed_params=None,
                default_params=None,
                eval_function=classification_error,
                model_init_func=None,
                max_evaluations=10):
    fn = build_fn(model_class,
                  eval_function,
                  X_train, X_valid,
                  y_train, y_valid,
                  default_params=default_params,
                  model_init_func=model_init_func)
    if allowed_params is None:
        params = model_class.params
    else:
        params = dict((p, model_class.params[p]) for p in allowed_params)
    if not_allowed_params is not None:
        for p in not_allowed_params:
            del params[p]
    all_params, all_scores = minimize_fn((fn, max_evaluations, params))

    for i, param in enumerate(all_params):
        all_params[i] = preprocessed_params_from_model_class(param, model_class)

    return all_params, all_scores


def find_best_hp(*args, **kwargs):
    all_params, all_scores = find_all_hp(*args, **kwargs)
    argmin = min(range(len(all_params)), key=lambda i: all_scores[i])
    parameters, loss = all_params[argmin], all_scores[argmin]
    return parameters, loss

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
    space = build_hp_space(parameter_definition)
    trials = Trials()
    result = fmin(fn, space, algo=tpe.suggest,
                  max_evals=max_evaluations,
                  trials=trials)
    for param_name, param_value in parameter_definition.items():
        if param_value.type == 'choice':
            result[param_name] = param_value.interval[result[param_name]]
    hyper_parameters = [r.get("hp")
                        for r in trials.results if r.get("status") == 'ok']
    return hyper_parameters, trials.losses()

def build_hp_space(parameter_definition):
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
                space[param_name] = hp.loguniform(param_name,
                                                  a * np.log(d),
                                                  b * np.log(d))
        elif param_value.type == 'int':
            a, b = param_value.interval
            if param_value.scale == 'normal':
                space[param_name] = hp.quniform(param_name, a, b, 1.)
            elif param_value.scale.startswith('log'):
                if len(param_value.scale) > len('log'):
                    d = int(param_value.scale[len('log'):])
                else:
                    d = np.exp(1)
                space[param_name] = hp.qloguniform(param_name,
                                                   a * np.log(d),
                                                   b * np.log(d))
        elif param_value.type == 'choice':
            space[param_name] = hp.choice(param_name, param_value.interval)
    return space


def incorporate_params(inst, params):
    for k, v in inst.params.items():
        setattr(inst, k, params.get(k, v.initial))
    inst.__dict__.update(params)


def instantiate_random(params, default_params=None, rng=None):
    params_inst = build_hp_space(params)
    params_inst = sample(params_inst, rng=rng)
    params_inst = preprocessed_params(params_inst, params)
    if default_params is not None:
        params_inst.update(default_params)
    return params_inst


def instantiate_random_model(model_class, default_params=None):
    params = instantiate_random(model_class.params, default_params=default_params)
    return model_class(**params)


def instantiate_default(params, default_params=None):
    params_inst = dict((name, param.initial)
                       for name, param in params.items())
    params_inst = preprocessed_params(params_inst, params)
    if default_params is not None:
        params_inst.update(default_params)
    return params_inst


def instantiate_default_model(model_class, default_params=None):
    params = instantiate_default(model_class.params, default_params=default_params)
    return model_class(**params)


class Model(object):

    def __init__(self, **params):
        self.state = 1234
        incorporate_params(self, params)
