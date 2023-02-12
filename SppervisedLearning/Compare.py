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
from HelperFunctions import custom_f1, f1_m
from keras.callbacks import EarlyStopping, ModelCheckpoint
from numpy.random import seed
import tensorflow as tf

x_train,y_train, x_test, y_test = getData1()


DT1= DecisionTreeClassifier(criterion='entropy',random_state=0,max_depth=3, max_leaf_nodes=3,ccp_alpha=.045)
SVM1 = SVC(random_state=42, kernel= 'linear', gamma=.05,C=.1)
boost1 = AdaBoostClassifier(n_estimators=100,learning_rate=1.8, random_state=42,estimator=DecisionTreeClassifier(random_state=0,max_depth=4)) #

knn1 =  KNeighborsClassifier(n_neighbors=10, weights='uniform')


models=[DT1,SVM1, boost1, knn1]

result=[]
for model in models:
    t1=time.time()
    clf = model.fit(x_train,y_train)
    pred=clf.predict(x_test)
    t2=time.time()
    delta=t2-t1
    result.append(model.__class__.__name__,recall_score(y_test, pred),delta)
result_pd=pd.DataFrame(result,columns=["model",'score','run_time'])







result_pd["dataset"] = 1


x_train,y_train, x_test, y_test = getData2()
result=[]
DT2 = DecisionTreeClassifier(criterion="entropy",random_state=0,max_depth=4, max_leaf_nodes=10,ccp_alpha=.0025)
SVM2 =SVC(random_state=42, gamma=1, kernel='linear')
knn2 = KNeighborsClassifier(n_neighbors=30, weights='uniform',metric='euclidean')
boost2 = AdaBoostClassifier(n_estimators=80,learning_rate=.8, random_state=42,estimator=DecisionTreeClassifier(random_state=0)) #

models=[DT2,SVM2, boost2, knn2]
x_train,y_train, x_test, y_test = getData1()
result=[]
for model in models:
    t1=time.time()
    clf = model.fit(x_train,y_train)
    pred=clf.predict(x_test)
    t2=time.time()
    delta=t2-t1
    result.append(model.__class__.__name__,recall_score(y_test, pred),delta)
result_pd=pd.DataFrame(result,columns=["model",'score','run_time'])

model = Sequential()
nodes=4
dropout="no "

model.add(Dense(nodes, input_dim=x_train.shape[1], activation='relu'))
# model.add(Dropout(dropout))

model.add(Dense(1))
model.add(Activation('sigmoid'))
opt = keras.optimizers.Adam()  #
model.compile(loss =tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=opt,  # also try adam
              metrics=["Recall"])  # f1_m
# class_weights = {0: .7, 1: 1}
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)

history = model.fit(x_train, y_train, verbose=1, epochs=1000, batch_size=64,callbacks=[es])
#
_, f1_score = model.evaluate(x_test, y_test, verbose=0)





result_pd["dataset"] = 2
