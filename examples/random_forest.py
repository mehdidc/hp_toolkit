from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split

from hp_toolkit.hp import minimize_fn_with_hyperopt, Param, find_best_hp


class RandomForestWrapper(RandomForestClassifier):

    params = dict(
            max_depth=Param(initial=5, interval=[1, 10], type='int'),
            n_estimators=Param(initial=50, interval=[10, 300], type='int')
    )


X, y = make_classification(n_samples=1000, n_features=20,
                           n_informative=2, n_redundant=2)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1)


def eval_function(model, X, y):
    return (model.predict(X) != y).mean()

best_hp = find_best_hp(RandomForestWrapper,
                       (minimize_fn_with_hyperopt),
                       X_train, X_valid,
                       y_train, y_valid,
                       eval_function=eval_function,
                       max_evaluations=20)
print(best_hp)
