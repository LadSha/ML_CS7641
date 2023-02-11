import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import LearningCurveDisplay, ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from IPython.display import display
from DataPrep import get_data
from HelperFunctions import prepare_val_curve
import pandas as pd

x_train,y_train, x_test, y_test = get_data()
metric = 'accuracy'


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


def KNN_experiment():

    model = KNeighborsClassifier()#metric='minkowski',weights='distance'

    prepare_val_curve(model,"n_neighbors",[30,50,100,150,175,200,300],metric,"KNN",x_train,y_train)


    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True)

    k=30
    result=[]
    columns=["k", "distance_metric", "test_accuracy", "train_accuracy", "fit_time",
                                              "score_time"]
    for distance_metric in ['manhattan', 'cosine', 'minkowski', 'euclidean']:
        model = KNeighborsClassifier(n_neighbors=k, metric=distance_metric,)
        cv_results = cross_validate(estimator=model, X=x_train, y=y_train, cv=skf, return_train_score=True,
                                    scoring=["accuracy"])
        metrics = {'mean_' + k: np.mean(v) for k, v in cv_results.items()}  # if k not in ["fit_time", "score_time"]
        print(metrics)
        result.append(
            [k, distance_metric, metrics["mean_test_accuracy"], metrics["mean_train_accuracy"], metrics["mean_fit_time"],
             metrics["mean_score_time"]])
    pd_result = pd.DataFrame(result, columns=columns)
    display(pd_result)


    columns = ["k", "weights", "test_accuracy", "train_accuracy", "fit_time",
               "score_time"]
    parameters = [[30, 'uniform'], [30, 'distance']]
    result=[]

    for k, weights in parameters:
        model = KNeighborsClassifier(n_neighbors=k, weights=weights)
        cv_results = cross_validate(estimator=model, X=x_train, y=y_train, cv=skf, return_train_score=True,
                                    scoring=["accuracy"])
        metrics = {'mean_' + k: np.mean(v) for k, v in cv_results.items()}  # if k not in ["fit_time", "score_time"]
        print(metrics)
        result.append(
            [k, weights, metrics["mean_test_accuracy"], metrics["mean_train_accuracy"], metrics["mean_fit_time"],
             metrics["mean_score_time"]])
    pd_result = pd.DataFrame(result, columns=columns)
    display(pd_result)



    # fig, ax = plt.subplots()
    # n = round(len(x_train) * .8)
    # common_params = {
    #     "X": x_train,
    #     "y": y_train,
    #     "train_sizes": np.arange(k, n-k, 500),
    #     "cv": ShuffleSplit(n_splits=50, test_size=0.2, random_state=0),
    #     "score_type": "both",
    #     "n_jobs": 4,
    #     "line_kw": {"marker": "o"},
    #     "std_display_style": "fill_between",
    #     "score_name": metric,
    # }
    #
    #
    # train_sizes, train_scores, valid_scores = learning_curve(neigh, x_train, y_train, cv=5, shuffle=True)
    # LearningCurveDisplay.from_estimator(neigh, **common_params, ax=ax)
    # handles, label = ax.get_legend_handles_labels()
    # ax.legend(handles, ["Training Score", "Test Score"])
    # ax.set_title(f"Learning Curve K={k}, weights=uniform {neigh.__class__.__name__}")
    # plt.show()
    #
    # neigh = KNeighborsClassifier(n_neighbors=k,weights='distance')
    #
    # fig, ax = plt.subplots()
    # n = round(len(x_train) * .8)
    # common_params = {
    #     "X": x_train,
    #     "y": y_train,
    #     "train_sizes": np.arange(k, n-k, 500),
    #     "cv": ShuffleSplit(n_splits=50, test_size=0.2, random_state=0),
    #     "score_type": "both",
    #     "n_jobs": 4,
    #     "line_kw": {"marker": "o"},
    #     "std_display_style": "fill_between",
    #     "score_name": metric,
    # }
    #
    #
    # train_sizes, train_scores, valid_scores = learning_curve(neigh, x_train, y_train, cv=5, shuffle=True)
    # LearningCurveDisplay.from_estimator(neigh, **common_params, ax=ax)
    # handles, label = ax.get_legend_handles_labels()
    # ax.legend(handles, ["Training Score", "Test Score"])
    # ax.set_title(f"Learning Curve K={k}, weights=distance {neigh.__class__.__name__}")
    # plt.show()
    k = 30
    neigh = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='euclidean')

    fig, ax = plt.subplots()
    n = round(len(x_train) * .8)
    common_params = {
        "X": x_train,
        "y": y_train,
        "train_sizes": np.arange(k, n - k, 500),
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
    ax.set_title(f"Learning Curve K={k}, weights=distance {neigh.__class__.__name__}")
    plt.show()


    k=50
    neigh = KNeighborsClassifier(n_neighbors=k,weights='distance',metric='euclidean')


    fig, ax = plt.subplots()
    n = round(len(x_train) * .8)
    common_params = {
        "X": x_train,
        "y": y_train,
        "train_sizes": np.arange(k, n - k, 500),
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
    ax.set_title(f"Learning Curve K={k}, weights=distance {neigh.__class__.__name__}")
    plt.show()

    k=70
    neigh = KNeighborsClassifier(n_neighbors=k,weights='distance',metric='euclidean')


    fig, ax = plt.subplots()
    n = round(len(x_train) * .8)
    common_params = {
        "X": x_train,
        "y": y_train,
        "train_sizes": np.arange(k, n - k, 500),
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
    ax.set_title(f"Learning Curve K={k}, weights=distance {neigh.__class__.__name__}")
    plt.show()



if __name__ == "__main__":
    KNN_experiment()
    neigh = KNeighborsClassifier(n_neighbors=30, metric='minkowski', weights='distance')

