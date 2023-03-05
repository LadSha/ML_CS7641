import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
import numpy as np
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LearningCurveDisplay, ShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.metrics import recall_score, f1_score
import time
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from DataPrep1 import get_data as getData1
from DataPrep2 import get_data as getData2
#Reference https://github.com/bnsreenu/python_for_microscopists/blob/master/154_understanding_train_validation_loss_curves.py
#Reference https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-k-fold-cross-validation-with-keras.md
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import seaborn as sns
from tensorflow import keras
from sklearn.model_selection import KFold
from DataPrep2 import get_data
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from HelperFunctions import  f1_m, get_val_scores
from keras.callbacks import EarlyStopping, ModelCheckpoint
from numpy.random import seed
import tensorflow as tf
from IPython.display import display


def first_dataset():
    x_train,y_train, x_test, y_test = getData1()
    metric="recall"

    DT1= DecisionTreeClassifier(criterion='entropy',random_state=0,max_depth=3, max_leaf_nodes=3,ccp_alpha=.045)
    SVM1 = SVC(random_state=42, kernel= 'linear', gamma=.05,C=.1)
    boost1 = AdaBoostClassifier(n_estimators=100,learning_rate=1.8, random_state=42,estimator=DecisionTreeClassifier(random_state=0,max_depth=4)) #

    knn1 =  KNeighborsClassifier(n_neighbors=10, weights='uniform')


    models=[DT1,SVM1, boost1, knn1]
    train_result=[]
    test_result=[]
    for model in models:
        t1=time.time()
        clf = model.fit(x_train,y_train)
        t2=time.time()
        pred=clf.predict(x_test)
        t3=time.time()

        test_result.append([model.__class__.__name__,recall_score(y_test, pred),t2-t1,t3-t2])
        metrics = get_val_scores(model,x_train,y_train,metric)
        train_result.append([model.__class__.__name__, metrics[f"mean_test_{metric}"], metrics[f"mean_train_{metric}"], metrics["mean_fit_time"],
             metrics["mean_score_time"]])

    X_tr, X_val, y_tr, y_val = train_test_split(x_train, y_train, test_size=0.15, random_state=0, stratify=y_train)

    t1=time.time()
    nodes = 4
    dropout = "no "
    model = Sequential()
    model.add(Dense(nodes, input_dim=X_tr.shape[1], activation='relu'))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    opt = keras.optimizers.Adam()  #
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,  # also try adam
                  metrics=["Recall"])  # f1_m
    # class_weights = {0: .7, 1: 1}
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)



    history = model.fit(X_tr, y_tr, verbose=1, epochs=500, batch_size=64,
                        validation_data=(X_val, y_val),callbacks=[es])
    t2=time.time()

    _, scor=model.evaluate(x_test,y_test)
    t3=time.time()

    train_result.append([model.__class__.__name__, np.mean(history.history["val_recall"]), np.mean(history.history["recall"]),
                         "-",
                         "-"])

    test_result.append([model.__class__.__name__,scor,t2-t1,t3-t2])

    result_pd_test=pd.DataFrame(test_result,columns=["model",'test_score',"train_time","score_time"])

    result_pd_test["dataset"] = 1
    print("    unseen data results")
    display(result_pd_test)

    print("    cross validation result")
    display(pd.DataFrame(train_result,
                             columns=["model", f"val_{metric}", f"train_{metric}", "fit_time", "score_time"]))


#second dataset
def second_dataset():
    x_train,y_train, x_test, y_test = getData2()
    metric="f1"

    DT2 = DecisionTreeClassifier(criterion="entropy",random_state=0,max_depth=4, max_leaf_nodes=10,ccp_alpha=.0025)
    SVM2 =SVC(random_state=42, gamma=1, kernel='linear')
    knn2 = KNeighborsClassifier(n_neighbors=30, weights='uniform',metric='euclidean')
    boost2 = AdaBoostClassifier(n_estimators=80,learning_rate=.8, random_state=42,estimator=DecisionTreeClassifier(random_state=0)) #

    models=[DT2,SVM2, boost2, knn2]
    test_result=[]
    train_result=[]
    for model in models:
        t1 = time.time()
        clf = model.fit(x_train, y_train)
        t2 = time.time()
        pred = clf.predict(x_test)
        t3 = time.time()

        test_result.append([model.__class__.__name__, f1_score(y_test, pred), t2 - t1, t3 - t2])
        metrics = get_val_scores(model, x_train, y_train, metric)
        train_result.append([model.__class__.__name__, metrics[f"mean_test_{metric}"], metrics[f"mean_train_{metric}"],
                             metrics["mean_fit_time"],
                             metrics["mean_score_time"]])
    X_tr, X_val, y_tr, y_val = train_test_split(x_train, y_train, test_size=0.15, random_state=0, stratify=y_train)

    t1 = time.time()
    model = Sequential()
    nodes = 4
    dropout = .16
    model.add(Dense(nodes, input_dim=X_tr.shape[1], activation='relu'))
    model.add(Dropout(dropout))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    opt = keras.optimizers.Adam()  #
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=opt,
                  metrics=[f1_m])  # f1_m



    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)

    history = model.fit(X_tr, y_tr, verbose=1, epochs=1000, batch_size=64,
                        validation_data=(X_val, y_val), callbacks=[es])
    t2=time.time()
    print(t2-t1)
    _, scor=model.evaluate(x_test,y_test)
    t3=time.time()

    train_result.append(
        [model.__class__.__name__, np.mean(history.history["val_f1_m"]), np.mean(history.history["f1_m"]),
         "-",
         "-"])

    test_result.append([model.__class__.__name__, scor, t2 - t1, t3 - t2])

    result_pd_test = pd.DataFrame(test_result, columns=["model", 'test_score', "train_time", "score_time"])

    result_pd_test["dataset"] = 2
    print("    unseen data results")
    display(result_pd_test)

    print("    cross validation result")
    display(pd.DataFrame(train_result,
                         columns=["model", f"val_{metric}", f"train_{metric}", "fit_time", "score_time"]))


if __name__=="__main__":
    seed(1)
    # first_dataset()
    second_dataset()

