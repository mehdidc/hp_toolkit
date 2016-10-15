def feed(feval, inputs, outputs):
    """
    decorate the hyperopt feval (fmin) function
    by a collection of pairs of inputs and outputs at the beginning.
    the motivation is to have a way to save hyperopt 'state' and load
    it later to continue later.
    """
    def feval_(x):
        if feval_.i < len(inputs):
            print(feval_.i)
            assert inputs[feval_.i] == x, 'Check the'
            output = outputs[feval_.i]
            feval_.i += 1
            return output
        else:
            return feval(x)
    feval_.i = 0
    return feval_


def get_from_trials(trials, name):
    return [t[name] for t in trials.trials]

if __name__ == '__main__':
    from hyperopt import hp, fmin, tpe, rand, Trials, STATUS_OK # NOQA
    import numpy as np

    space = hp.uniform('a', 0, 1)

    def fn(x):
        print(x)
        return {'loss': x**2-4, 'status': STATUS_OK, 'x': x}

    np.random.seed(11)
    trials = Trials()
    best = fmin(
        fn=fn,
        space=space,
        algo=rand.suggest,
        rseed=1,
        max_evals=10,
        trials=trials)

    result = get_from_trials(trials, 'result')

    inputs = [r['x'] for r in result]
    outputs = result
    print(inputs)

    trials = Trials()
    best = fmin(fn=feed(fn, inputs, outputs),
                space=space, algo=rand.suggest, max_evals=15,
                rseed=2,
                trials=trials)

    result = get_from_trials(trials, 'result')
    inputs = [r['x'] for r in result]
    outputs = result
    print(inputs)
