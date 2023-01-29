import DT
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
import numpy as np
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LearningCurveDisplay, ShuffleSplit
from sklearn.model_selection import learning_curve

def prepare_data(dataset_name):
    dataset = pd.read_csv(f'../Data/{dataset_name}')
    y_train = dataset['class']
    x_train = dataset.drop(labels=['class'], axis=1)

    le = LabelEncoder()

    cols = x_train.columns.values
    for col in cols:
        x_train[col] = le.fit_transform(x_train[col])

    y_train = le.fit_transform(y_train)

    ohe = OneHotEncoder()
    x_train = ohe.fit_transform(x_train).toarray()
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    return x_train, y_train

# x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.30, random_state=42)
# x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test, test_size=0.50, random_state=42)
##K-fold
# model = DecisionTreeClassifier()
# n_folds=5
# skf = StratifiedKFold(n_splits=n_folds, shuffle=True)
# cv_results = cross_validate (estimator=model, X=x_train, y=y_train, cv=skf,scoring=["accuracy", "f1", "precision", "recall", "roc_auc"])
# metrics = {'mean_' + k: np.mean(v) for k, v in cv_results.items() if k not in ["fit_time", "score_time"]}
#
# print(x_train)
# print(cv_results)

# ##Validation curve
def prepare_val_curve(param_name,param_range,scoring, algorithm_name):
    x_train, y_train = prepare_data()
    train_scores, test_scores = validation_curve(
        DecisionTreeClassifier(random_state=0),
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

# ##Grid search
def grid_search(parameters, scoring, refit, model=DecisionTreeClassifier(random_state=0)):
    x_train, y_train = prepare_data()

    parameters={'max_depth':np.arange(1,40,1), 'criterion':('gini','entropy','log_loss'), 'min_samples_leaf':np.arange(1,10,1)}
    clf = GridSearchCV(model, parameters,scoring='f1', refit='f1')
    clf.fit(x_train,y_train)
    x=(pd.DataFrame(clf.cv_results_))
    x.to_excel('grid.xlsx')
    print(clf.best_score_)
    print(clf.best_params_)

#learning curve
train_sizes, train_scores, valid_scores = learning_curve(DecisionTreeClassifier(random_state=0), x_train, y_train, cv=5)

model=DecisionTreeClassifier(random_state=0)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 6), sharey=True)

common_params = {
    "X": x_train,
    "y": y_train,
    "train_sizes": np.linspace(0.1, 1.0, 9),
    "cv": ShuffleSplit(n_splits=50, test_size=0.2, random_state=0),
    "score_type": "both",
    "n_jobs": 4,
    "line_kw": {"marker": "o"},
    "std_display_style": "fill_between",
    "score_name": "f1",
}

for ax_idx, estimator in enumerate([model]):
    LearningCurveDisplay.from_estimator(estimator, **common_params, ax=ax[ax_idx])
    handles, label = ax[ax_idx].get_legend_handles_labels()
    ax[ax_idx].legend(handles[:2], ["Training Score", "Test Score"])
    ax[ax_idx].set_title(f"Learning Curve for {estimator.__class__.__name__}")
plt.show()

def perform_experiments():
