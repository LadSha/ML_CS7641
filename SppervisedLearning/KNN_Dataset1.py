import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import LearningCurveDisplay, ShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.neighbors import KNeighborsClassifier

from DataPrep import get_data
from HelperFunctions import prepare_val_curve

x_train, y_train = get_data()
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

    train_sizes, train_scores, valid_scores = learning_curve(neigh, x_train, y_train, cv=5, shuffle=True)
    LearningCurveDisplay.from_estimator(neigh, **common_params, ax=ax)
    handles, label = ax.get_legend_handles_labels()
    ax.legend(handles, ["Training Score", "Test Score"])
    ax.set_title(f"Learning Curve K={k}, weights={weights} {neigh.__class__.__name__}")
    plt.show()


def KNN_experiment():
    # GS suggested  distance: {'algorithm': 'auto', 'metric': 'minkowski', 'n_neighbors': 2, 'weights': 'distance'}
    # # parameters = {'n_neighbors': np.arange(7, 45, 2) , 'weights':['uniform', 'distance'], 'metric':['manhattan','minkowski','euclidean'], 'algorithm':['auto', 'ball_tree','kd_tree','brute'],'p':np.arange(1,10,1)}
    #
    # # best_parm = grid_search(parameters, scoring=metric, refit=metric, model=neigh)
    #
    k=13
    for distance_metric in ['manhattan', 'cosine', 'minkowski', 'euclidean']:
        neigh = KNeighborsClassifier(n_neighbors=k, metric=distance_metric,)
        prepare_val_curve(neigh, "n_neighbors", np.arange(1, 65, 2), metric, f"KNN-{distance_metric}")

    parameters = [[13, 'uniform'], [13, 'distance'], [30, 'uniform'],[30, 'distance']]

    for k, weights in parameters:
        neigh = KNeighborsClassifier(n_neighbors=k, weights=weights)
        create_learning_curve(k, neigh, weights)
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


# pruning
# k=30
# neigh = KNeighborsClassifier(n_neighbors=k,weights='uniform')
#
# prepare_val_curve(neigh, "n_neighbors", np.arange(1, 65, 2), metric, "KNN-minkowski")
#
# fig, ax = plt.subplots()
# n = round(len(x_train) * .8)
# common_params = {
#     "X": x_train,
#     "y": y_train,
#     "train_sizes": np.arange(k, n - k, 500),
#     "cv": ShuffleSplit(n_splits=50, test_size=0.2, random_state=0),
#     "score_type": "both",
#     "n_jobs": 4,
#     "line_kw": {"marker": "o"},
#     "std_display_style": "fill_between",
#     "score_name": metric,
# }
#
# train_sizes, train_scores, valid_scores = learning_curve(neigh, x_train, y_train, cv=5, shuffle=True)
# LearningCurveDisplay.from_estimator(neigh, **common_params, ax=ax)
# handles, label = ax.get_legend_handles_labels()
# ax.legend(handles, ["Training Score", "Test Score"])
# ax.set_title(f"Learning Curve K={k}, weights=uniform {neigh.__class__.__name__}")
# plt.show()
#
# k = 30
# neigh = KNeighborsClassifier(n_neighbors=k, weights='distance')
#
# prepare_val_curve(neigh, "n_neighbors", np.arange(1, 65, 2), metric, "KNN-minkowski")
#
# fig, ax = plt.subplots()
# n = round(len(x_train) * .8)
# common_params = {
#     "X": x_train,
#     "y": y_train,
#     "train_sizes": np.arange(k, n - k, 500),
#     "cv": ShuffleSplit(n_splits=50, test_size=0.2, random_state=0),
#     "score_type": "both",
#     "n_jobs": 4,
#     "line_kw": {"marker": "o"},
#     "std_display_style": "fill_between",
#     "score_name": metric,
# }
#
# train_sizes, train_scores, valid_scores = learning_curve(neigh, x_train, y_train, cv=5, shuffle=True)
# LearningCurveDisplay.from_estimator(neigh, **common_params, ax=ax)
# handles, label = ax.get_legend_handles_labels()
# ax.legend(handles, ["Training Score", "Test Score"])
# ax.set_title(f"Learning Curve K={k}, weights=distance {neigh.__class__.__name__}")
# plt.show()

if __name__ == "__main__":
    KNN_experiment()
