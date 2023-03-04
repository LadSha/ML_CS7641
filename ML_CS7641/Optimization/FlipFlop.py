import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose_hiive
import numpy as np
from mlrose_hiive.runners import RHCRunner, SARunner, GARunner, MIMICRunner
from mlrose_hiive import TSPGenerator,FlipFlopGenerator,FourPeaks
import matplotlib.pyplot as plt
from time import time



def experiment(problem, problem_name):
    SEED=42
    rhc = RHCRunner(problem=problem,
                    experiment_name="RHC",
                    output_directory=f"/home/ladan/Desktop/Georgia Tech/ML_CS7641/ML_CS7641/Optimization/{problem_name}",
                    seed=SEED,
                    iteration_list=2 ** np.arange(10),
                    max_attempts=5000,
                    restart_list=[0,5, 20])

    rhc_run_stats, rhc_run_curves = rhc.run()

    fig, axes = plt.subplots()
    plt.plot(rhc_run_curves[rhc_run_curves["Restarts"]==0]["Fitness"].values, label=f"restarts=0")
    axes.set_xlabel("Iterations")
    axes.set_ylabel("Fitness Score")
    axes.set_title("RHC - Fitness vs Iterations for no. of restarts ({})".format(problem_name))
    axes.legend(loc="best")
    axes.grid()
    plt.savefig(f"/home/ladan/Desktop/Georgia Tech/ML_CS7641/ML_CS7641/Optimization/{problem_name}/rhc1.png")
    plt.show()
    fig, axes = plt.subplots()
    plt.plot(rhc_run_curves[rhc_run_curves["Restarts"]==5]["Fitness"].values, label=f"restarts=5")
    axes.set_xlabel("Iterations")
    axes.set_ylabel("Fitness Score")
    axes.set_title("RHC - Fitness vs Iterations for different no. of restarts ({})".format(problem_name))
    axes.legend(loc="best")
    axes.grid()
    plt.savefig(f"/home/ladan/Desktop/Georgia Tech/ML_CS7641/ML_CS7641/Optimization/{problem_name}/rhc2.png")
    plt.show()
    fig, axes = plt.subplots()
    plt.plot(rhc_run_curves[rhc_run_curves["Restarts"]==20]["Fitness"].values, label=f"restarts=20")
    axes.set_xlabel("Iterations")
    axes.set_ylabel("Fitness Score")
    axes.set_title("RHC - Fitness vs Iterations for different no. of restarts ({})".format(problem_name))
    axes.legend(loc="best")
    axes.grid()
    plt.savefig(f"/home/ladan/Desktop/Georgia Tech/ML_CS7641/ML_CS7641/Optimization/{problem_name}/rhc3.png")
    plt.show()

    n=2000
    sa = SARunner(problem=problem,
                         experiment_name="SA_final",
                         output_directory=f"/home/ladan/Desktop/Georgia Tech/ML_CS7641/ML_CS7641/Optimization/{problem_name}",
                         seed=SEED,
                         iteration_list=2 ** np.arange(13),
                         max_attempts=n,
                         temperature_list=[0.01, 0.1, 5, 10],
                         decay_list=[mlrose_hiive.ExpDecay])

    sa_run_stats, sa_run_curves = sa.run()
    fig, axes = plt.subplots()
    plt.plot(sa_run_curves["Fitness"][:n].values, label=f"temp=0.01")
    plt.plot(sa_run_curves["Fitness"][2*n:3*n].values, label=f"temp=0.1")
    plt.plot(sa_run_curves["Fitness"][3*n:4*n].values, label=f"temp=5")
    plt.plot(sa_run_curves["Fitness"][4*n:5*n].values, label=f"temp=10")
    axes.set_xlabel("Iterations")
    axes.set_ylabel("Fitness Score")
    axes.set_title("SA - Fitness vs Iterations for different temp ({})".format(problem_name))
    axes.legend(loc="best")
    axes.grid()
    plt.savefig(f"/home/ladan/Desktop/Georgia Tech/ML_CS7641/ML_CS7641/Optimization/{problem_name}/sa.png")
    plt.show()

    ga = GARunner(problem=problem,
                         experiment_name="GA_final",
                         output_directory=f"/home/ladan/Desktop/Georgia Tech/ML_CS7641/ML_CS7641/Optimization/{problem_name}",
                         seed=SEED,
                         iteration_list=2 ** np.arange(13),
                         max_attempts=1000,
                         population_sizes=[5, 100, 200],
                         mutation_rates=[0.4,.6,0.3])
    ga_run_stats, ga_run_curves = ga.run()

    fig, axes = plt.subplots()
    plt.plot(ga_run_curves[(ga_run_curves["Population Size"] == 5) & (ga_run_curves["Mutation Rate"] == 0.4)][
                 "Fitness"].values, label=f"Population Size=5")

    plt.plot(ga_run_curves[(ga_run_curves["Population Size"] == 100) & (ga_run_curves["Mutation Rate"] == 0.4)][
                 "Fitness"].values, label=f"Population Size=100")

    plt.plot(ga_run_curves[(ga_run_curves["Population Size"] == 200) & (ga_run_curves["Mutation Rate"] == 0.4)][
                 "Fitness"].values, label=f"Population Size=200")
    axes.set_xlabel("Iterations")
    plt.xlim(left=0,right=1000)
    axes.set_ylabel("Fitness Score")
    axes.set_title("GA - Fitness vs Iterations for different mutate rate ({})".format(problem_name))
    axes.legend(loc="best")
    axes.grid()
    plt.savefig(f"/home/ladan/Desktop/Georgia Tech/ML_CS7641/ML_CS7641/Optimization/{problem_name}/ga1.png")
    plt.show()

    fig, axes = plt.subplots()
    plt.plot(ga_run_curves[(ga_run_curves["Population Size"] == 200) & (ga_run_curves["Mutation Rate"] == 0.4)][
                 "Fitness"].values, label=f"mutate_rate=0.4")

    plt.plot(ga_run_curves[(ga_run_curves["Population Size"] == 200) & (ga_run_curves["Mutation Rate"] == 0.6)][
                 "Fitness"].values, label=f"mutate_rate=0.6")

    plt.plot(ga_run_curves[(ga_run_curves["Population Size"] == 200) & (ga_run_curves["Mutation Rate"] == 0.3)][
                 "Fitness"].values, label=f"mutate_rate=0.3")
    axes.set_xlabel("Iterations")
    plt.xlim(left=0,right=1000)
    axes.set_ylabel("Fitness Score")
    axes.set_title("GA - Fitness vs Iterations for different pop_size ({})".format(problem_name))
    axes.legend(loc="best")
    axes.grid()
    plt.savefig(f"/home/ladan/Desktop/Georgia Tech/ML_CS7641/ML_CS7641/Optimization/{problem_name}/ga2.png")
    plt.show()


    mmc = MIMICRunner(problem=problem,
                      experiment_name="MIMIC_final",
                      output_directory=f"/home/ladan/Desktop/Georgia Tech/ML_CS7641/ML_CS7641/Optimization/{problem_name}",
                      seed=SEED,
                      iteration_list=2 ** np.arange(10),
                      population_sizes=[200,300],
                      max_attempts=500,
                      keep_percent_list=[0.25,.5,.75],
                      use_fast_mimic=True)
    mimic_run_stats, mimic_run_curves = mmc.run()

    fig, axes = plt.subplots()
    plt.plot(mimic_run_curves[(mimic_run_curves["Population Size"] == 200) & (mimic_run_curves["Keep Percent"] == 0.25)][
                 "Fitness"].values, label="Keep %=0.25")
    plt.plot(mimic_run_curves[(mimic_run_curves["Population Size"] == 200) & (mimic_run_curves["Keep Percent"] == 0.5)][
                 "Fitness"].values, label="Keep %=0.5")
    plt.plot(mimic_run_curves[(mimic_run_curves["Population Size"] == 200) & (mimic_run_curves["Keep Percent"] == 0.75)][
                 "Fitness"].values, label="Keep %=0.75")
    axes.set_xlabel("Iterations")
    axes.set_ylabel("Fitness Score")
    axes.set_title("MIMIC - Fitness vs Iterations for different keep % ({})".format(problem_name))
    axes.legend(loc="best")
    axes.grid()
    plt.savefig(f"/home/ladan/Desktop/Georgia Tech/ML_CS7641/ML_CS7641/Optimization/{problem_name}/mmc1.png")
    plt.show()

    fig, axes = plt.subplots()
    plt.plot(mimic_run_curves[(mimic_run_curves["Population Size"] == 300) & (mimic_run_curves["Keep Percent"] == 0.25)][
                 "Fitness"].values, label="pop_size=300")
    plt.plot(mimic_run_curves[(mimic_run_curves["Population Size"] == 200) & (mimic_run_curves["Keep Percent"] == 0.25)][
                 "Fitness"].values, label="pop_size=200")
    axes.set_xlabel("Iterations")
    axes.set_ylabel("Fitness Score")
    axes.set_title("MIMIC - Fitness vs Iterations for different keep % ({})".format(problem_name))
    axes.legend(loc="best")
    axes.grid()
    plt.savefig(f"/home/ladan/Desktop/Georgia Tech/ML_CS7641/ML_CS7641/Optimization/{problem_name}/mmc2.png")
    plt.show()


if __name__=="__main__":
    SEED = 42
    problem_name = 'FilpFlop-prob_size=100'
    problem = FlipFlopGenerator.generate(seed=27, size=100)
    experiment(problem,problem_name)

    problem_name = 'FilpFlop-prob_size=500'
    problem = FlipFlopGenerator.generate(seed=27, size=500)
    experiment(problem,problem_name)