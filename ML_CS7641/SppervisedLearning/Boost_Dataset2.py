import matplotlib.pyplot as plt
from DataPrep2 import get_data
from sklearn.model_selection import train_test_split
import numpy as np
from HelperFunctions import grid_search, prepare_val_curve,create_learning_curve
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

x_train,y_train, x_test, y_test = get_data()
metric = 'f1'


def experiment():

    n=80
    lr=.8
    modelDT = AdaBoostClassifier(random_state=0)#

    classfier_name="AdaBoost-defaultParams"
    prepare_val_curve(modelDT,'learning_rate',[.1,.5,.75,1, 1.5, 2, 2.5],metric,classfier_name,x_train,y_train)
    prepare_val_curve(modelDT,'n_estimators',np.arange(5,100,4),metric,classfier_name,x_train,y_train)

    modelDT = AdaBoostClassifier(n_estimators=n,learning_rate=lr, random_state=0)
    create_learning_curve(modelDT, metric, f"{n}estmatrs-{lr}-LearnRate-no leaf/depth limit", x_train, y_train)

    modelDT = AdaBoostClassifier(n_estimators=n,learning_rate=lr, random_state=0, estimator=DecisionTreeClassifier(random_state=0,max_depth=5, max_leaf_nodes=5))
    create_learning_curve(modelDT, metric, f"{n}estimators-{lr}LearnRate-5depth-5leaves", x_train, y_train)

    modelDT = AdaBoostClassifier(n_estimators=50,learning_rate=lr, random_state=0, estimator=DecisionTreeClassifier(random_state=0,max_depth=3, max_leaf_nodes=3))

    create_learning_curve(modelDT, metric, f"{50}estimators-{lr}LearnRate-3depth-3leaves", x_train, y_train)

    n=100
    lr=.7
    modelDT = AdaBoostClassifier(n_estimators=n,learning_rate=lr, random_state=0, estimator=DecisionTreeClassifier(random_state=0,max_depth=3, max_leaf_nodes=3))

    create_learning_curve(modelDT, metric, f"{n}estimators-{lr}LearnRate-3depth-3leaves", x_train, y_train)


def GrSearch():
    modelDT = AdaBoostClassifier(n_estimators=50, random_state=0)
    parameters = {'n_estimators': np.arange(1, 50, 1),
                  'learning_rate': [0.1, .25, 0.5,.75, 1,1.25, 1.5, 2, 2.5]}  # , 'ccp_alpha': np.arange(0, .05, 0.01).tolist()
    best_parm = grid_search(parameters, scoring=metric, refit=metric, model=modelDT, x_train=x_train, y_train=y_train)
    print(best_parm)
    return best_parm


if __name__ == '__main__':
    # GrSearch()
    experiment()

