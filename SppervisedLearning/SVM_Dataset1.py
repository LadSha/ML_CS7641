import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import LearningCurveDisplay, ShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
from DataPrep import get_data
from HelperFunctions import prepare_val_curve
from HelperFunctions import create_learning_curve

x_train, y_train = get_data()
metric = 'accuracy'





def SVM_experiment():

    svc = SVC(random_state=42)

    parameters=[['gamma',[.0001, 0.001, 0.01, 1]]] #],

    # for param, param_range in parameters:
    #     prepare_val_curve(svc,param,param_range, metric,f"SVM")

    for kernel in [ 'linear', 'poly', 'rbf','sigmoid']:#'
        model=SVC(random_state=42, kernel= kernel, gamma=.008,tol=.1)
        create_learning_curve(model,metric,f'tol=.1 kernel={kernel}')

    for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
        model=SVC(random_state=42, kernel= kernel, gamma=.008, tol=.1, class_weight='balanced')
        create_learning_curve(model,metric,f'tol=.1,balancedWeight, kernel={kernel}')




if __name__ == "__main__":
    SVM_experiment()
    SVC(random_state=42, gamma=.008, tol=.1,class_weight=None, kernel='linear')