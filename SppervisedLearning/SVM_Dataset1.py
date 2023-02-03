import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import LearningCurveDisplay, ShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
from HelperFunctions import prepare_val_curve

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
        "train_sizes": np.arange(k, n - k, 500),
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


def SVM_experiment():

    svc = SVC(random_state=42)

    parameters=[['gamma',np.arange(.001,.01,.001)], ]

    for param, param_range in parameters:
        prepare_val_curve(svc,param,param_range, metric,f"SVM")

    for kernel in ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']:
        model=SVC(random_state=42, kernel= kernel)
        prepare_val_curve(model,)





    for distance_metric in ['manhattan', 'cosine', 'minikowski', 'euclidean']:
        neigh = KNeighborsClassifier(n_neighbors=k, metric=distance_metric)
        prepare_val_curve(neigh, "n_neighbors", np.arange(1, 65, 2), metric, f"KNN-{distance_metric}")


if __name__ == "__main__":
    KNN_experiment()
