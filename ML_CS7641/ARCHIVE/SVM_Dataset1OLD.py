import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import LearningCurveDisplay, ShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
from DataPrep import get_data
from HelperFunctions import prepare_val_curve
from HelperFunctions import create_learning_curve
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate, StratifiedKFold
from IPython.display import display

x_train,y_train, x_test, y_test = get_data()
metric = 'accuracy'


def SVM_experiment():
    svc = SVC(random_state=42)

    parameters=[['gamma',[.0001, 0.001, .003,.006,.008, 0.01, 1]]] #],

    for param, param_range in parameters:
        prepare_val_curve(svc,param,param_range, metric,f"SVM",x_train,y_train)

    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True)
    result=[]

    for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:  # '
        model = SVC(random_state=42, kernel=kernel, gamma=.008)
        cv_results = cross_validate(estimator=model, X=x_train, y=y_train, cv=skf, return_train_score=True,
                                    scoring=["accuracy", "f1", "precision", "recall", "roc_auc"])
        metrics = {'mean_' + k: np.mean(v) for k, v in cv_results.items()}  # if k not in ["fit_time", "score_time"]
        print(metrics)
        result.append([kernel, 0.001, metrics["mean_test_accuracy"],metrics["mean_train_accuracy"], metrics["mean_fit_time"], metrics["mean_score_time"]])
    pd_result = pd.DataFrame(result, columns=["kernel","tolerance", "test_accuracy","train_accuracy", "fit_time", "score_time"])
    display(pd_result)

    for kernel in [ 'linear', 'poly', 'rbf','sigmoid']:#'
        model=SVC(random_state=42, kernel= kernel, gamma=.008,tol=.1)
        cv_results = cross_validate(estimator=model, X=x_train, y=y_train, cv=skf, return_train_score=True,
                                    scoring=["accuracy", "f1", "precision", "recall", "roc_auc"])
        metrics = {'mean_' + k: np.mean(v) for k, v in cv_results.items() } #if k not in ["fit_time", "score_time"]
        print(metrics)
        result.append([kernel,0.1,metrics["mean_test_accuracy"],metrics["mean_train_accuracy"],metrics["mean_fit_time"], metrics["mean_score_time"]])
    pd_result= pd.DataFrame(result,columns= ["kernel","tolerance", "test_accuracy","train_accuracy", "fit_time","score_time"])
    display(pd_result)


    for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
        model=SVC(random_state=42, kernel= kernel, gamma=.008)
        create_learning_curve(model,metric,f' kernel={kernel}',x_train,y_train)


    for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
        model=SVC(random_state=42, kernel= kernel, gamma=.008, class_weight='balanced')
        create_learning_curve(model,metric,f'balancedWeight, kernel={kernel}',x_train,y_train)




if __name__ == "__main__":
    SVM_experiment()
    SVC(random_state=42, gamma=.008, tol=.1,class_weight=None, kernel='linear')