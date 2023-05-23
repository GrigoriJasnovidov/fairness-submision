from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier

from fmclp.core import lp_solver, predictor, ml_model
from fmclp.synthetic_dataset import synthetic_dataset
from fmclp.cuae_metric import cuae


def fmclp(dataset, estimator, number_iterations=None, prefit=False, interior_classifier='rf',
          verbose=False, multiplier=1, random_state=None):
    interior_classifier_dict = {'rf': RandomForestClassifier(random_state=random_state),
                                'lr': LogisticRegression(),
                                'dt': DecisionTreeClassifier(random_state=random_state),
                                'svm': SVC(),
                                'lgb': LGBMClassifier(),
                                'knn': KNeighborsClassifier(n_neighbors=3)}

    model = ml_model(df=dataset, estimator=estimator, random_state=random_state, prefit=prefit)
    solved = lp_solver(model,
                       classifier=interior_classifier_dict[interior_classifier],
                       number_iterations=number_iterations,
                       verbose=verbose,
                       multiplier=multiplier,
                       random_state=random_state)
    pred = predictor(solved, model, verbose)
    fair_cuae = cuae(y_true=model['x_test'], y_pred=pred['preds'],
                     sensitive_features=model['y_test']['attr'])
    fair_accuracy = accuracy_score(pred['preds'], model['x_test'])

    ans = {'accuracy_of_initial_classifier': model['estimator_accuracy'],
           'fairness_of_initial_classifier': cuae(y_true=model['x_test'], y_pred=model['predictions'],
                                                  sensitive_features=model['y_test']['attr']),
           'solved': solved,
           'predictions': pred,
           'fairness_of_fair_classifier': fair_cuae,
           'accuracy_of_fair_classifier': fair_accuracy,
           'dataset': dataset,
           'model': model,
           'multiplier': multiplier,
           'interior_classifier': interior_classifier
           }

    return ans


# example 
if __name__ == '__main__':
    d = synthetic_dataset(400)
    dataset = synthetic_dataset(20000)
    random_state = 78
    cl = LGBMClassifier(random_state=random_state)
    y = d.drop('target', axis=1)
    x = d['target']
    y_train, y_test, x_train, x_test = train_test_split(y, x, random_state=random_state)
    cl.fit(y_train, x_train)
    res = fmclp(dataset=dataset,
                estimator=cl,
                number_iterations=10,
                prefit=True,
                interior_classifier='knn',
                verbose=True,
                multiplier=30,
                random_state=random_state)
