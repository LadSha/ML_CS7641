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
from sklearn import neural_network

from sklearn.datasets import make_classification

x_train, y_train = make_classification(n_samples=8113, n_features=10, random_state=42)
# x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=0)

# def prepare_data(dataset_name):
# dataset = pd.read_csv(f'../Data/mushrooms.csv')
# y_train = dataset['class']
# x_train = dataset.drop(labels=['class'], axis=1)
#
# le = LabelEncoder()
#
# cols = x_train.columns.values
# for col in cols:
#     x_train[col] = le.fit_transform(x_train[col])
#
# y_train = le.fit_transform(y_train)
#
# ohe = OneHotEncoder()
# x_train = ohe.fit_transform(x_train).toarray()
# sc = StandardScaler()
# x_train = sc.fit_transform(x_train)
#
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

# ##Grid search
def grid_search(parameters, scoring, refit, model=DecisionTreeClassifier(random_state=0)):


    clf = GridSearchCV(model, parameters,scoring=scoring, refit=refit, error_score='raise')
    clf.fit(x_train,y_train)
    x=(pd.DataFrame(clf.cv_results_))
    x.to_excel('grid.xlsx')
    print(clf.best_params_)
    return(clf.best_params_.values)

#learning curve
def create_learning_curve(models,metric):


    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 6), sharey=True)
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

    for ax_idx, estimator in enumerate(models):
        train_sizes, train_scores, valid_scores = learning_curve(estimator, x_train, y_train, cv=5, shuffle=True)
        LearningCurveDisplay.from_estimator(estimator, **common_params, ax=ax[ax_idx])
        handles, label = ax[ax_idx].get_legend_handles_labels()
        ax[ax_idx].legend(handles[:2], ["Training Score", "Test Score"])
        ax[ax_idx].set_title(f"Learning Curve for {estimator.__class__.__name__}")
    plt.show()

if __name__=="__main__":
    #Decision Tree

#find alpha
#     clf = DecisionTreeClassifier(random_state=0)
#     path = clf.cost_complexity_pruning_path(x_train, y_train)
#     ccp_alphas, impurities = path.ccp_alphas, path.impurities
#
#     fig, ax = plt.subplots()
#     ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
#     ax.set_xlabel("effective alpha")
#     ax.set_ylabel("total impurity of leaves")
#     ax.set_title("Total Impurity vs effective alpha for training set")
#     clfs = []
#     for ccp_alpha in ccp_alphas:
#         clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
#         clf.fit(x_train, y_train)
#         clfs.append(clf)
#     print(
#         "Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
#             clfs[-1].tree_.node_count, ccp_alphas[-1]
#         )
#     )
#     clfs = clfs[:-1]
#     ccp_alphas = ccp_alphas[:-1]
#
#     node_counts = [clf.tree_.node_count for clf in clfs]
#     depth = [clf.tree_.max_depth for clf in clfs]
#     fig, ax = plt.subplots(2, 1)
#     ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
#     ax[0].set_xlabel("alpha")
#     ax[0].set_ylabel("number of nodes")
#     ax[0].set_title("Number of nodes vs alpha")
#     ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
#     ax[1].set_xlabel("alpha")
#     ax[1].set_ylabel("depth of tree")
#     ax[1].set_title("Depth vs alpha")
#     fig.tight_layout()
#
#     train_scores = [clf.score(x_train, y_train) for clf in clfs]
#     test_scores = [clf.score(x_test, y_test) for clf in clfs]
#
#     fig, ax = plt.subplots()
#     ax.set_xlabel("alpha")
#     ax.set_ylabel("accuracy")
#     ax.set_title("Accuracy vs alpha for training and testing sets")
#     ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
#     ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
#     ax.legend()
#     plt.show()
#
#
# #grid search
#
#     metric="accuracy"
#     modelDT=DecisionTreeClassifier(random_state=0,ccp_alpha=.003)
#     parameters = {'max_depth': np.arange(1, 40, 1), 'criterion': ('gini', 'entropy', 'log_loss'),
#                   'max_leaf_nodes': np.arange(2, 40, 1)} #, 'ccp_alpha': np.arange(0, .05, 0.01).tolist()
#     best_parm = grid_search(parameters, scoring=metric, refit=metric, model=modelDT)
#
#     prepare_val_curve('max_depth',np.arange(1,40,1),metric,"DTClassifier")
#     prepare_val_curve('max_leaf_nodes', np.arange(1, 40, 1), metric, "DTClassifier")
#
#
#
#
#
#     modelDT=DecisionTreeClassifier(random_state=0,max_depth=5, max_leaf_nodes=10)
#     create_learning_curve([modelDT], metric)
    #ANN
    metric = "accuracy"
    modelNN=neural_network.MLPClassifier(random_state=0,hidden_layer_sizes = (100,100,),max_iter=3000)
    modelNN.fit(x_train,y_train)
    parameters={
    #'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
    # 'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],}

    # grid_search(parameters,metric,metric,modelNN)

    prepare_val_curve(modelNN, 'learning_rate_init', param_range=[100,100],scoring=metric,algorithm_name="MLPClassifier")



