from collections import OrderedDict
from tabulate import tabulate

from sklearn.datasets import fetch_covtype
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cross_validation import train_test_split

from hp_toolkit.hp import minimize_fn_with_hyperopt
from hp_toolkit.hp import Param
from hp_toolkit.hp import find_best_hp, find_all_hp
from hp_toolkit.hp import incorporate_params

from sklearn.datasets import fetch_mldata, fetch_covtype
from sklearn.utils import shuffle

class RandomForestWrapper(RandomForestClassifier):

    params = dict(
            max_depth=Param(initial=5, interval=[1, 50], type='int'),
            n_estimators=Param(initial=50, interval=[10, 800], type='int'),
            criterion=Param(initial="gini",
                            interval=["gini", "entropy"], type='choice'),
            max_leaf_nodes=Param(initial=10, interval=[10, 200], type='int')
    )

class GradientBoostingWrapper(GradientBoostingClassifier):

    params = dict(
        loss=Param(initial='deviance', interval=["deviance", "exponential"], type='choice'),
        n_estimators=Param(initial=50, interval=[10, 800], type='int'),
        max_leaf_nodes=Param(initial=10, interval=[10, 200], type='int'),
        learning_rate = Param(initial=10e-4, interval=[-5, 0], type='real', scale='log10'),
    )

#wrapper = RandomForestWrapper
#default_params = dict(n_jobs=-1)

wrapper = GradientBoostingWrapper
default_params = dict()


#data = fetch_mldata('MNIST original')
data = fetch_covtype()

print("data : {0}".format(data))
print("wrapper : {0}".format(wrapper))

X = data['data']
y = data['target']

X, y = shuffle(X, y)
print(X.shape, y.shape)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25)


def classification_error(model, X, y):
    return (model.predict(X) != y).mean()

best_hp = find_best_hp(wrapper,
                       minimize_fn_with_hyperopt,
                       X_train, X_valid,
                       y_train, y_valid,
                       eval_function=classification_error,
                       max_evaluations=10,
                       default_params=default_params)
print(best_hp)

all_hp, all_scores = find_all_hp(RandomForestWrapper,
                                 minimize_fn_with_hyperopt,
                                 X_train, X_valid,
                                 y_train, y_valid,
                                 eval_function=classification_error,
                                 max_evaluations=200,
                                 default_params=default_params)
ranking = sorted(range(len(all_hp)), key=lambda index: all_scores[index])
all_hp = [all_hp[r] for r in ranking]
all_scores = [all_scores[r] for r in ranking]
for hp, score in zip(all_hp, all_scores):
    output = OrderedDict()
    output.update({k: [v] for k, v in hp.items()})
    output["score"] = [score]
    print(tabulate(output, headers="keys"))
