import numpy as np
import matplotlib.pyplot as plt
from utils import save_figure


def plot_rhc(path):
    data = np.load(path + "optimization/rhc_fitness_curve.npy")

    plt.figure()
    plt.plot(data, color='b')
    plt.title("Neural Network With RHC Algorithm \n Learning Curve")
    plt.xlabel("Iterations")
    plt.ylabel("Loss Error")
    plt.grid()
    save_figure(plt, path + "plot/optimization/", "rhc_learning_curve.png")
    # plt.show()


def plot_sa(path):
    data = np.load(path + "optimization/sa_fitness_curve.npy")

    plt.figure()
    plt.plot(data, color='b')
    plt.title("Neural Network With Simulated Annealing Algorithm \n Learning Curve")
    plt.xlabel("Iterations")
    plt.ylabel("Loss Error")
    plt.grid()
    save_figure(plt, path + "plot/optimization/", "sa_learning_curve.png")
    # plt.show()


def plot_ga(path):
    data = np.load(path + "optimization/ga_fitness_curve.npy")

    plt.figure()
    plt.plot(data, color='b')
    plt.title("Neural Network With Genetic Algorithm \n Learning Curve")
    plt.xlabel("Iterations")
    plt.ylabel("Loss Error")
    plt.grid()
    save_figure(plt, path + "plot/optimization/", "ga_learning_curve.png")
    # plt.show()


def plot_sgd(path):
    data = np.load(path + "optimization/sgd_fitness_curve.npy")

    plt.figure()
    plt.plot(data, color='b')
    plt.title("Neural Network With SGD algorithm \n Learning Curve")
    plt.xlabel("Iterations")
    plt.ylabel("Loss Error")
    plt.grid()
    save_figure(plt, path + "plot/optimization/", "sgd_learning_curve.png")
    # plt.show()


def plot_model(path, algorithm):

    if algorithm == 'rhc':
        plot_rhc(path)

    if algorithm == 'ga':
        plot_sa(path)

    if algorithm == 'sa':
        plot_ga(path)

    if algorithm == 'sgd':
        plot_sgd(path)

    if algorithm == 'all':
        plot_rhc(path)
        plot_sa(path)
        plot_ga(path)
        plot_sgd(path)


if __name__ == "__main__":
    plot_model('../', 'all')

