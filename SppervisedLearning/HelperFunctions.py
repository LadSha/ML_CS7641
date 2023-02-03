
import numpy as np
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LearningCurveDisplay, ShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn import neural_network
import time
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from DataPrep import get_data

x_train, y_train = get_data()
def prepare_val_curve(model,param_name,param_range,scoring, algorithm_name):

    train_scores, test_scores = validation_curve(
        model,
        x_train,
        y_train,
        param_name=param_name,
        param_range=param_range,
        scoring=scoring
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title(f"Validation Curve with {algorithm_name}")
    plt.xlabel(f"{param_name}")
    plt.ylabel(f"{scoring}")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.semilogx(
        param_range, train_scores_mean, label="Training score", color="darkorange", lw=lw
    )
    plt.fill_between(
        param_range,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.2,
        color="darkorange",
        lw=lw,
    )
    plt.semilogx(
        param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=lw
    )
    plt.fill_between(
        param_range,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.2,
        color="navy",
        lw=lw,
    )
    plt.legend(loc="best")
    plt.show()

def create_learning_curve(estimator,metric,title):

#reference: https://scikit-learn.org/
    fig, ax = plt.subplots()
    n=round(len(x_train)*.8)
    common_params = {
        "X": x_train,
        "y": y_train,
        "train_sizes": np.arange(1,n,500),
        "cv": ShuffleSplit(n_splits=50, test_size=0.2, random_state=0),
        "score_type": "both",
        "n_jobs": 4,
        "line_kw": {"marker": "o"},
        "std_display_style": "fill_between",
        "score_name": metric,
    }


    train_sizes, train_scores, valid_scores = learning_curve(estimator, x_train, y_train, cv=5, shuffle=True)
    LearningCurveDisplay.from_estimator(estimator, **common_params, ax=ax)
    handles, label = ax.get_legend_handles_labels()
    ax.legend(handles[:2], ["Training Score", "Test Score"])
    ax.set_title(f"Learning Curve for {estimator.__class__.__name__} {title}")
    plt.show()

def grid_search(parameters, scoring, refit, model):


    clf = GridSearchCV(model, parameters,scoring=scoring, refit=refit, error_score='raise')
    clf.fit(x_train,y_train)
    print(clf.best_params_)
    return(clf.best_params_.values)
