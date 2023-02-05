import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import LearningCurveDisplay, ShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
from DataPrep2 import get_data
from HelperFunctions import prepare_val_curve
from HelperFunctions import create_learning_curve
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate, StratifiedKFold
from IPython.display import display
from sklearn.model_selection import validation_curve


x_train,y_train, x_test, y_test = get_data()
metric = 'f1'


def SVM_experiment():
    svc = SVC(random_state=42)

    parameters=[['gamma',[ .006, 0.01,.05,.08, .1,.2,.3,.5,1,5]]] #],

    for param, param_range in parameters:
        prepare_val_curve(svc,param,param_range, metric,f"SVM", x_train,y_train)

    gamma=.1
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True)
    result=[]

    for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:  # '
        model = SVC(random_state=42, kernel=kernel, gamma=gamma)
        cv_results = cross_validate(estimator=model, X=x_train, y=y_train, cv=skf, return_train_score=True,
                                    scoring=[metric])
        metrics = {'mean_' + k: np.mean(v) for k, v in cv_results.items()}  # if k not in ["fit_time", "score_time"]
        print(metrics)
        result.append([kernel, 0.001, metrics[f"mean_test_{metric}"],metrics[f"mean_train_{metric}"], metrics["mean_fit_time"], metrics["mean_score_time"]])
    pd_result = pd.DataFrame(result, columns=["kernel","tolerance", f"test_{metric}",f"train_{metric}", "fit_time", "score_time"])
    display(pd_result)

    for kernel in [ 'linear', 'poly', 'rbf','sigmoid']:#'
        model=SVC(random_state=42, kernel= kernel, gamma=gamma,tol=.1)
        cv_results = cross_validate(estimator=model, X=x_train, y=y_train, cv=skf, return_train_score=True,
                                    scoring=[metric])
        metrics = {'mean_' + k: np.mean(v) for k, v in cv_results.items() } #if k not in ["fit_time", "score_time"]
        print(metrics)
        result.append([kernel,0.1,metrics[f"mean_test_{metric}"],metrics[f"mean_train_{metric}"],metrics["mean_fit_time"], metrics["mean_score_time"]])
    pd_result= pd.DataFrame(result,columns= ["kernel","tolerance", f"test_{metric}",f"train_{metric}", "fit_time","score_time"])
    display(pd_result)


    for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
        model=SVC(random_state=42, kernel= kernel, gamma=gamma)
        create_learning_curve(model,metric,f' kernel={kernel}', x_train,y_train)


    for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
        model=SVC(random_state=42, kernel= kernel, gamma=gamma, class_weight='balanced')
        create_learning_curve(model,metric,f'balancedWeight, kernel={kernel}',x_train,y_train)




if __name__ == "__main__":
    SVM_experiment()
#     SVC(random_state=42, gamma=.008, tol=.1,class_weight=None, kernel='linear')
# svc=SVC(random_state=42, gamma=.008, tol=.1,class_weight=None, kernel='linear')
#
# print(validation_curve(estimator=svc,X=x_train,y=y_train,param_name='kernel',param_range= ['linear','poly'],cv=5,scoring=metric))