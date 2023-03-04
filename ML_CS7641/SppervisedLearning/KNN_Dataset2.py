import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import LearningCurveDisplay, ShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from IPython.display import display
from DataPrep2 import get_data
from HelperFunctions import prepare_val_curve,grid_search
import pandas as pd
from sklearn.model_selection import GridSearchCV


x_train,y_train, x_test, y_test = get_data()
metric = 'f1'


def create_learning_curve(k, neigh, weights):
    fig, ax = plt.subplots()
    n = round(len(x_train) * .8)
    common_params = {
        "X": x_train,
        "y": y_train,
        "train_sizes": np.arange(k, n - k, 100),
        "cv": ShuffleSplit(n_splits=50, test_size=0.2, random_state=0),
        "score_type": "both",
        "n_jobs": 4,
        "line_kw": {"marker": "o"},
        "std_display_style": "fill_between",
        "score_name": metric,
    }

    LearningCurveDisplay.from_estimator(neigh, **common_params, ax=ax)
    handles, label = ax.get_legend_handles_labels()
    ax.legend(handles, ["Training Score", "Test Score"])
    ax.set_title(f"Learning Curve K={k}, weights={weights} {neigh.__class__.__name__}")
    plt.show()


def experiment():
    # GS suggested  distance: {'algorithm': 'auto', 'metric': 'minkowski', 'n_neighbors': 2, 'weights': 'distance'}
    # # parameters = {'n_neighbors': np.arange(7, 45, 2) , 'weights':['uniform', 'distance'], 'metric':['manhattan','minkowski','euclidean'], 'algorithm':['auto', 'ball_tree','kd_tree','brute'],'p':np.arange(1,10,1)}
    # #
    # # # best_parm = grid_search(parameters, scoring=metric, refit=metric, model=neigh)
    # #
    #
    distance_metric='euclidean'
    weights='distance'
    model = KNeighborsClassifier(weights=weights,metric=distance_metric)#metric='minkowski',weights='distance'
    prepare_val_curve(model,"n_neighbors",[5,10,15,20,25,30,35],metric,f"KNN-{weights}",x_train,y_train)

    weights = 'uniform'
    model = KNeighborsClassifier(weights=weights,metric=distance_metric)  # metric='minkowski',weights='distance'
    prepare_val_curve(model, "n_neighbors", [5,10, 15, 20, 25, 30, 35,50,100,130], metric, f"KNN-{weights}", x_train, y_train)

    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True)

    # clf = GridSearchCV(model, {"n_neighbors":[2,5,8,9,10,11,12,15,17,20,25,30,35,45,55]},scoring=metric, error_score='raise')
    # clf.fit(x_train,y_train)
    # print(clf.best_params_)

    k=30
    result=[]
    columns=["k", "distance_metric", f"test_{metric}", f"train_{metric}", "fit_time",
                                              "score_time"]
    for distance_metric in ['manhattan', 'cosine', 'minkowski', 'euclidean']:
        model = KNeighborsClassifier(n_neighbors=k, metric=distance_metric,weights=weights)
        cv_results = cross_validate(estimator=model, X=x_train, y=y_train, cv=skf, return_train_score=True,
                                    scoring=[metric])
        metrics = {'mean_' + k: np.mean(v) for k, v in cv_results.items()}  # if k not in ["fit_time", "score_time"]
        print(metrics)
        result.append(
            [k, distance_metric, metrics[f"mean_test_{metric}"], metrics[f"mean_train_{metric}"], metrics["mean_fit_time"],
             metrics["mean_score_time"]])
    pd_result = pd.DataFrame(result, columns=columns)
    display(pd_result)


    columns = ["k", "weights", f"test_{metric}", f"train_{metric}", "fit_time",
               "score_time"]
    parameters = [[k, 'uniform'], [k, 'distance']]
    result=[]

    for k, w in parameters:
        model = KNeighborsClassifier(n_neighbors=k, weights=w)
        cv_results = cross_validate(estimator=model, X=x_train, y=y_train, cv=skf, return_train_score=True,
                                    scoring=[metric])
        metrics = {'mean_' + k: np.mean(v) for k, v in cv_results.items()}  # if k not in ["fit_time", "score_time"]
        print(metrics)
        result.append(
            [k, w, metrics[f"mean_test_{metric}"], metrics[f"mean_train_{metric}"], metrics["mean_fit_time"],
             metrics["mean_score_time"]])
    pd_result = pd.DataFrame(result, columns=columns)
    display(pd_result)

    k = 10
    neigh = KNeighborsClassifier(n_neighbors=k, weights=weights, metric=distance_metric)
    create_learning_curve(k, neigh, weights)

    k = 30
    neigh = KNeighborsClassifier(n_neighbors=k, weights=weights, metric=distance_metric)
    create_learning_curve(k, neigh, weights)
    k = 50
    neigh = KNeighborsClassifier(n_neighbors=k, weights=weights, metric=distance_metric)
    create_learning_curve(k, neigh, weights)

    k = 60
    neigh = KNeighborsClassifier(n_neighbors=k, weights=weights, metric=distance_metric)
    create_learning_curve(k, neigh, weights)




if __name__ == "__main__":
    experiment()
    neigh = KNeighborsClassifier(n_neighbors=50, metric='euclidean', weights='uniform')

