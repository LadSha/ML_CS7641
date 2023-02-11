from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from DataPrep4 import get_data
from sklearn.model_selection import train_test_split
import numpy as np
from HelperFunctions import grid_search, prepare_val_curve,create_learning_curve

x_train,y_train, x_test, y_test = get_data()
metric = 'recall'



def DT_experiement():
    modelDT=DecisionTreeClassifier(criterion='entropy',random_state=0)
    # find_alph( modelDT)
    # GrSearch()

    prepare_val_curve(modelDT,'max_depth',np.arange(1,40,1),metric,"DTClassifier-Default Params",x_train,y_train)
    prepare_val_curve(modelDT, 'max_leaf_nodes', np.arange(1, 40, 1), metric, "DTClassifier-Default Params",x_train,y_train)
    create_learning_curve(modelDT, metric, "DT-default", x_train, y_train)
    # find_alph( modelDT)

    max_depth=3
    max_leaf=3
    alpha=.045
    modelDT=DecisionTreeClassifier(criterion = 'entropy',random_state=0,max_depth=max_depth, max_leaf_nodes=max_leaf)
    find_alph( modelDT)

    create_learning_curve(modelDT, metric, f"DT-maxDepth={max_depth} leaf_nodes={max_leaf}", x_train, y_train)
    modelDT=DecisionTreeClassifier(random_state=0,max_depth=max_depth, max_leaf_nodes=max_leaf,ccp_alpha=alpha,criterion='entropy')
    create_learning_curve(modelDT, metric, f"DT-maxDepth={max_depth} leaf_nodes={max_leaf} alpha={alpha}", x_train, y_train)


# #find alpha
def find_alph(clf):
    #Reference: https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html#sphx-glr-auto-examples-tree-plot-cost-complexity-pruning-py

    path = clf.cost_complexity_pruning_path(x_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities

    fig, ax = plt.subplots()
    ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
    ax.set_xlabel("effective alpha")
    ax.set_ylabel("total impurity of leaves")
    ax.set_title("Total Impurity vs effective alpha for training set")
    clfs = []
    for ccp_alpha in ccp_alphas:
        clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
        clf.fit(x_train, y_train)
        clfs.append(clf)
    print(
        "Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
            clfs[-1].tree_.node_count, ccp_alphas[-1]
        )
    )
    clfs = clfs[:-1]
    ccp_alphas = ccp_alphas[:-1]

    node_counts = [clf.tree_.node_count for clf in clfs]
    depth = [clf.tree_.max_depth for clf in clfs]
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
    ax[0].set_xlabel("alpha")
    ax[0].set_ylabel("number of nodes")
    ax[0].set_title("Number of nodes vs alpha")
    ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
    ax[1].set_xlabel("alpha")
    ax[1].set_ylabel("depth of tree")
    ax[1].set_title("Depth vs alpha")
    fig.tight_layout()
    x_train2, x_test2, y_train2, y_test2 = train_test_split(x_train, y_train, test_size=0.30, random_state=42)

    train_scores = [clf.score(x_train2, y_train2) for clf in clfs]

    test_scores = [clf.score(x_test2, y_test2) for clf in clfs]

    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs alpha for training and testing sets")
    ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
    ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
    ax.legend(loc="lower left")
    plt.show()
#
# #
# # #grid search
def GrSearch():
    modelDT=DecisionTreeClassifier(random_state=0,max_depth=3, max_leaf_nodes=3,ccp_alpha=.03)
    parameters = { 'criterion': ('gini', 'entropy', 'log_loss'),
                  } #, 'ccp_alpha': np.arange(0, .05, 0.01).tolist()
    best_parm = grid_search(parameters, scoring=metric, refit=metric, model=modelDT,x_train=x_train,y_train=y_train)
    print(best_parm)
    return best_parm

if __name__=='__main__':

    DT_experiement()
    # GrSearch()
    modelDT=DecisionTreeClassifier(criterion='entropy',random_state=0,max_depth=3, max_leaf_nodes=3,ccp_alpha=.045)

