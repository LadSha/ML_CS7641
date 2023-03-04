from mlrose_hiive import ArithDecay
from mlrose_hiive.runners import NNGSRunner
import mlrose_hiive
from SppervisedLearning.DataPrep2 import get_data

x_train,y_train, x_test, y_test = get_data()
def nn_model():
    grid_search_parameters = ({
        'max_iters': [1, 2, 4, 8, 16, 32, 64, 128],                     # nn params
        'learning_rate': [0.001, 0.002, 0.003],                         # nn params
        'schedule': [ArithDecay(1), ArithDecay(100), ArithDecay(1000)]  # sa params
    })
    nnr = NNGSRunner(x_train=x_train,
                     y_train=y_train,
                     x_test=x_test,
                     y_test=y_test,
                     experiment_name='nn_test',
                     algorithm=mlrose_hiive.algorithms.sa.simulated_annealing,
                     grid_search_parameters=grid_search_parameters,
                     iteration_list=[1, 10, 50, 100, 250, 500, 1000, 2500, 5000, 10000],
                     hidden_layer_sizes=[[44,44]],
                     bias=True,
                     early_stopping=False,
                     clip_max=1e+10,
                     max_attempts=500,
                     generate_curves=True,
                     seed=200972)
    results = nnr.run()

if __name__=="__main__":
    print(nn_model())