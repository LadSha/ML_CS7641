import matplotlib.pyplot as plt
from DataPrep1 import get_data
from sklearn.model_selection import train_test_split
import numpy as np
from HelperFunctions import grid_search, prepare_val_curve,create_learning_curve
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std
from matplotlib import pyplot
from numpy import arange


metric = 'recall'



def experiment():
	# def get_models_nestimators():
	# 	models = dict()
	# 	# define number of trees to consider
	# 	n_trees = [10, 50, 100, 500, 1000, 5000]
	# 	for n in n_trees:
	# 		models[str(n)] = AdaBoostClassifier(n_estimators=n, random_state=0)
	# 	return models
	#
	# # evaluate a given model using cross-validation
	# def evaluate_model(model, X, y):
	# 	# define the evaluation procedure
	# 	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	# 	# evaluate the model and collect the results
	# 	scores = cross_val_score(model, X, y, scoring=metric, cv=cv, n_jobs=-1)
	# 	return scores
	#
	# # define dataset
	X, y , x_test, y_test = get_data()
	# # get the models to evaluate
	# models = get_models_nestimators()
	# # evaluate the models and store results
	# results, names = list(), list()
	# for name, model in models.items():
	# 	# evaluate the model
	# 	scores = evaluate_model(model, X, y)
	# 	# store the results
	# 	results.append(scores)
	# 	names.append(name)
	# 	# summarize the performance along the way
	# 	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
	# # plot model performance for comparison
	# pyplot.boxplot(results, labels=names, showmeans=True)
	# pyplot.xlabel('n_estimators')
	# pyplot.ylabel(metric)
	# pyplot.show()
	#
	# def get_models_treeDepth():
	# 	models = dict()
	# 	# explore depths from 1 to 10
	# 	for i in range(1,11):
	# 		# define base model
	# 		base = DecisionTreeClassifier(max_depth=i,random_state=0)
	# 		# define ensemble model
	# 		models[str(i)] = AdaBoostClassifier(n_estimators=100,estimator=base)
	# 	return models
	#
	#
	# models = get_models_treeDepth()
	# # evaluate the models and store results
	# results, names = list(), list()
	# for name, model in models.items():
	# 	# evaluate the model
	# 	scores = evaluate_model(model, X, y)
	# 	# store the results
	# 	results.append(scores)
	# 	names.append(name)
	# 	# summarize the performance along the way
	# 	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
	#
	# pyplot.boxplot(results, labels=names, showmeans=True)
	# pyplot.xlabel('tree_depth')
	# pyplot.ylabel(metric)
	# pyplot.show()
	#
	# def get_models_learnRate():
	# 	models = dict()
	# 	# explore learning rates from 0.1 to 2 in 0.1 increments
	# 	for i in arange(0.1, 2.1, 0.1):
	# 		key = '%.3f' % i
	# 		models[key] = AdaBoostClassifier(learning_rate=i,n_estimators=100, estimator=DecisionTreeClassifier(max_depth=4,random_state=0))
	# 	return models
	#
	# models = get_models_learnRate()
	# # evaluate the models and store results
	# results, names = list(), list()
	# for name, model in models.items():
	# 	# evaluate the model
	# 	scores = evaluate_model(model, X, y)
	# 	# store the results
	# 	results.append(scores)
	# 	names.append(name)
	# 	# summarize the performance along the way
	# 	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
	#
	# pyplot.boxplot(results, labels=names, showmeans=True)
	# pyplot.xlabel('learning_rate')
	# plt.xticks(rotation=90)
	# pyplot.ylabel(metric)
	# pyplot.show()
	#
	#
	x_train,y_train = X, y
	n=100
	lr=1.8
	# modelDT = AdaBoostClassifier(random_state=0)#
	# #
	# classfier_name="AdaBoost-defaultParams"
	# prepare_val_curve(modelDT,'learning_rate',[.5,.75,1, 1.5, 1.8, 2, 2.5],metric,classfier_name,x_train,y_train)
	# prepare_val_curve(modelDT,'n_estimators',[1,10, 50, 100, 500, 1000, 5000],metric,classfier_name,x_train,y_train)
	# #
	# modelDT = AdaBoostClassifier(n_estimators=n,learning_rate=lr, random_state=0)
	# create_learning_curve(modelDT, metric, f"{n}estmatrs-{lr}-LearnRate-no leaf/depth", x_train, y_train)
	#
	modelDT = AdaBoostClassifier(n_estimators=n,learning_rate=lr, random_state=42,estimator=DecisionTreeClassifier(random_state=0,max_depth=4)) #
	# #
	create_learning_curve(modelDT, metric, f"{n}estimators-{lr}-4maxDepth", x_train, y_train)

if __name__ == '__main__':
    # GrSearch()
    experiment()

