from collections import OrderedDict
from tabulate import tabulate

from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split

from hp_toolkit.hp import minimize_fn_with_hyperopt
from hp_toolkit.hp import Param
from hp_toolkit.hp import find_best_hp, find_all_hp


class RandomForestWrapper(RandomForestClassifier):

    params = dict(
            max_depth=Param(initial=5, interval=[1, 10], type='int'),
            n_estimators=Param(initial=50, interval=[10, 500], type='int'),
            criterion=Param(initial="gini",
                            interval=["gini", "entropy"], type='choice'),
            max_leaf_nodes=Param(initial=10, interval=[10, 200], type='int')
    )


data = fetch_california_housing()
X = data['data']
y = data['target']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25)


def classification_error(model, X, y):
    return (model.predict(X) != y).mean()

best_hp = find_best_hp(RandomForestWrapper,
                       minimize_fn_with_hyperopt,
                       X_train, X_valid,
                       y_train, y_valid,
                       eval_function=classification_error,
                       max_evaluations=10)
print(best_hp)

all_hp, all_scores = find_all_hp(RandomForestWrapper,
                                 minimize_fn_with_hyperopt,
                                 X_train, X_valid,
                                 y_train, y_valid,
                                 eval_function=classification_error,
                                 max_evaluations=200)
ranking = sorted(range(len(all_hp)), key=lambda index: all_scores[index])
all_hp = [all_hp[r] for r in ranking]
all_scores = [all_scores[r] for r in ranking]
for hp, score in zip(all_hp, all_scores):
    output = OrderedDict()
    output.update({k: [v] for k, v in hp.items()})
    output["score"] = [score]
    print(tabulate(output, headers="keys"))
