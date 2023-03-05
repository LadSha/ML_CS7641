#Reference https://github.com/nikolasavic/randomized_optimization/blob/master/Simulated%20Annealing%20Grid%20Search.ipynb

from mlrose_hiive import ArithDecay
import six
import sys
sys.modules['sklearn.externals.six'] = six
from mlrose_hiive import NNGSRunner
import mlrose_hiive
from SppervisedLearning.DataPrep2 import get_data
from mlrose_hiive import NeuralNetwork
import mlrose
from time import time
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras.layers import Dense, LSTM, Conv1D
from tensorflow.python.keras.models import Sequential
import mlrose_hiive as mlrh
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from seglearn.datasets import load_watch
from seglearn.pipe import Pype
from seglearn.transform import Segment

x_train,y_train, x_test, y_test = get_data()

# def nn_model():

result=[]

# grid_search = {
#     "max_iters": [5000],
#     "learning_rate_init": [ .5],
#     "activation": [mlrh.neural.activation.relu],
#     "is_classifier": [True],
#     "temperature_list" :[.01]
# }
#
# t0=time()
# runner = mlrose_hiive.NNGSRunner(x_train=x_train, y_train=y_train,
#                            x_test=x_test, y_test=y_test,
#                            experiment_name="NN_SA",
#                            output_directory="nn_simulated_annealing/",
#                            algorithm=mlrose_hiive.algorithms.sa.simulated_annealing,
#                            grid_search_parameters=grid_search,
#                            iteration_list = [1, 10, 50, 100, 250, 500, 1000, 2500, 5000, 10000],
#                            hidden_layer_sizes=[[4]],
#                            bias=True,
#                            early_stopping=True,
#                            clip_max=1,
#                            max_attempts=1000,
#                            generate_curves=True,
#                            seed=42,
#                            n_jobs=-1
#                           )
# run_stats, curves, cv_results, best_est = runner.run()
# t1=time()
# print(best_est)
# fig, axes = plt.subplots()
# plt.plot(curves["Fitness"].values, label=f"SA-NN")
# axes.set_xlabel("Iterations")
# axes.set_ylabel("Loss")
# axes.set_title("Loss vs Iterations ({})".format("SA_NN"))
# axes.legend(loc="best")
# axes.grid()
# plt.savefig(f"/home/ladan/Desktop/Georgia Tech/ML_CS7641/ML_CS7641/Optimization/nn_genetic_algorithm/SA-loss.png")
# result.append(["SA_NN",t1-t0])
# plt.show()
#
# grid_search = {
#     "max_iters": [1000],
#     "activation": [mlrose_hiive.neural.activation.relu],
#     "is_classifier": [True],
#
# }
#
# t0=time()
# runner = mlrose_hiive.NNGSRunner(x_train=x_train, y_train=y_train,
#                            x_test=x_test, y_test=y_test,
#                            experiment_name="full_grid_search",
#                            output_directory="nn_randomized_hill_climbing/",
#                            algorithm=mlrose_hiive.algorithms.random_hill_climb,
#                            grid_search_parameters=grid_search,
#                            iteration_list = [1, 10, 50, 100, 250, 500, 1000, 2500, 5000, 10000],
#                            hidden_layer_sizes=[[4]],
#                            bias=True,
#                            early_stopping=True,
#                            clip_max=1,
#                            max_attempts=100,
#                            generate_curves=True,
#                            seed=42,
#                            n_jobs=-1
#                           )
# run_stats, curves, cv_results, best_est = runner.run()
# t1=time()
# fig, axes = plt.subplots()
# plt.plot(curves["Fitness"].values, label=f"RHC-NN")
# axes.set_xlabel("Iterations")
# axes.set_ylabel("Loss")
# axes.set_title("Loss vs Iterations ({})".format("RHC_NN"))
# axes.legend(loc="best")
# axes.grid()
# plt.savefig(f"/home/ladan/Desktop/Georgia Tech/ML_CS7641/ML_CS7641/Optimization/nn_genetic_algorithm/RHC-loss.png")
# result.append(["RHC_NN",t1-t0])
# plt.show()

grid_search = {
    "max_iters": [ 1000],
    "learning_rate_init": [0.1],
    "activation": [mlrh.neural.activation.relu],
    "is_classifier": [True],
    "mutation_prob": [0.25],
    "pop_size": [400]
}

t0=time()
runner = mlrh.NNGSRunner(x_train=x_train, y_train=y_train,
                           x_test=x_test, y_test=y_test,
                           experiment_name="full_grid_search",
                           output_directory="nn_genetic_algorithm/",
                           algorithm=mlrh.algorithms.genetic_alg,
                           grid_search_parameters=grid_search,
                           iteration_list = [1, 10, 50, 100, 250, 500, 1000, 2500, 5000, 10000],
                           hidden_layer_sizes=[[6,6]],
                           bias=True,
                           early_stopping=True,
                           clip_max=1,
                           max_attempts=1000,
                           generate_curves=True,
                           seed=42,
                           n_jobs=-1
                          )
run_stats, curves, cv_results, best_est = runner.run()
t1=time()

fig, axes = plt.subplots()
plt.plot(curves["Fitness"].values, label=f"GA-NN")
axes.set_xlabel("Iterations")
axes.set_ylabel("Loss")
axes.set_title("Loss vs Iterations ({})".format("GA_NN"))
axes.legend(loc="best")
axes.grid()
plt.savefig(f"/home/ladan/Desktop/Georgia Tech/ML_CS7641/ML_CS7641/Optimization/nn_genetic_algorithm/GA-loss.png")
result.append(["GA_NN",t1-t0])
plt.show()
print(result)
result.to_excel(f"/home/ladan/Desktop/Georgia Tech/ML_CS7641/ML_CS7641/Optimization/NN_results.xlsx")

# nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [11,4,1], activation = 'relu', \
#                                  algorithm = 'random_hill_climb', max_iters = 1000, \
#                                  bias = True, is_classifier = True, learning_rate = 0.0001, \
#                                  early_stopping = True, clip_max = 5, max_attempts = 100, \
#                                  random_state = 3)
# pipe = Pype([('seg', Segment(width=100, step=100, order='C')),('crnn', nn_model1)])
#
# pipe.fit(x_train, y_train)
# history = pipe.history.history
# print(DataFrame(history))

# nn_model1.fit(x_train, y_train)

# Predict labels for train set and assess accuracy
# y_pred = nn_model1.predict(x_test)
#
# y_train_f1 = f1_score(y_test, y_pred)
#
# print(y_train_f1)

# nn_model_rhc = mlrose_hiive.NeuralNetwork(hidden_nodes=[4], activation='relu',
#                                     algorithm='random_hill_climb', max_iters=1000,
#                                     bias=True, is_classifier=True, learning_rate=1,
#                                     early_stopping=True, clip_max=5, max_attempts=100,
#                                     random_state=0,curve=True)
# nn_model_sa = mlrose.NeuralNetwork(hidden_nodes=[4], activation='relu',
#                                    algorithm='simulated_annealing', max_iters=1000,
#                                    bias=True, is_classifier=True, learning_rate=1,
#                                    early_stopping=True, clip_max=5, max_attempts=100,
#                                    random_state=0,curve=True)
# nn_model_ga = mlrose.NeuralNetwork(hidden_nodes=[4], activation='relu',
#                                    algorithm='genetic_alg', max_iters=1000,
#                                    bias=True, is_classifier=True, learning_rate=1,
#                                    early_stopping=True, clip_max=5, max_attempts=100,
#                                    random_state=0,curve=True)
#
#
# nn_model_rhc.fit(x_train,y_train)
# y_pred = nn_model_rhc.predict(x_test)
# fit_curve = nn_model_rhc.fitness_curve
# plt.plot(fit_curve[:,0])
# plt.xlabel("Iterations")
# plt.ylabel("Loss")
# plt.title("NN_RHC")
# plt.show()
#
# nn_model_sa.fit(x_train,y_train)
# y_pred = nn_model_sa.predict(x_test)
# fit_curve = nn_model_sa.fitness_curve
# print(fit_curve)
# plt.plot(fit_curve)
# plt.xlabel("Iterations")
# plt.ylabel("Loss")
# plt.title("NN_SA")
# plt.show()
#
# nn_model_ga.fit(x_train,y_train)
# y_pred = nn_model_ga.predict(x_test)
# fit_curve = nn_model_ga.fitness_curve
# plt.plot(fit_curve)
# plt.xlabel("Iterations")
# plt.ylabel("Loss")
# plt.title("NN_GA")
# plt.show()