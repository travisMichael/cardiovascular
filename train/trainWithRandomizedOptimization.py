# Resources
# https://buildmedia.readthedocs.org/media/pdf/mlrose/stable/mlrose.pdf
from utils import load_data, save_figure, calculate_f1_score
from mlrosevariation import neural, decay
from sklearn.metrics import log_loss
import numpy as np



def train_neural_net_with_loan_data(path, with_plots):
    data_set = 'loan'
    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')
    X_test, y_test = load_data(path + 'data/' + data_set + '/test/')

    if with_plots == "False":
        # random_hill_climb
        # simulated_annealing
        # genetic_alg
        # nn_model1 = neural.NeuralNetwork(hidden_nodes = [], activation = 'sigmoid',
        #                                  algorithm = 'genetic_alg', max_iters = 1, pop_size=500, mutation_prob=0.3,
        #                                  bias = True, is_classifier = True, learning_rate = 0.001,
        #                                  early_stopping = True, clip_max = 1, max_attempts =100, random_state = 3)

        # genetic_alg
        decay_object = decay.ExpDecay(init_temp=.00001, exp_const=0.0005, min_temp=0.00000001)

        nn_model1 = neural.NeuralNetwork(hidden_nodes = [40], activation ='sigmoid',
                                         algorithm = 'simulated_annealing', max_iters = 5000, schedule=decay_object,
                                         bias = True, is_classifier = True, learning_rate = 0.5, curve=True,
                                         early_stopping = True, clip_max = 1, max_attempts =10, random_state = 3)

        print("Training")
        #
        print("Training")
        nn_model1.fit(x_train, y_train)
        # print(nn_model1.fitted_weights)
        np.save("sa_fitness_curve", nn_model1.fitness_curve)
        # y_estimate = nn_model1.predict(X_test)
        calculate_f1_score(nn_model1, x_train, y_train)
        calculate_f1_score(nn_model1, X_test, y_test)

        y_pred = nn_model1.predict_proba(X_test)

        loss = log_loss(y_test, y_pred)
        print(loss)


        # model_1 = train_and_time(MLPClassifier(solver='sgd', validation_fraction=0.0, alpha=1e-1, hidden_layer_sizes=(20, 5), random_state=1), x_train, y_train)
        # model_2 = train_and_time(MLPClassifier(solver='sgd', validation_fraction=0.0, alpha=1e-1, hidden_layer_sizes=(50, 5), random_state=1), x_train, y_train)
        # model_3 = train_and_time(MLPClassifier(solver='sgd', validation_fraction=0.0, alpha=1e-1, hidden_layer_sizes=(100, 5), random_state=1), x_train, y_train)
        # model_4 = train_and_time(MLPClassifier(solver='sgd', validation_fraction=0.0, alpha=1e-1, hidden_layer_sizes=(500, 5), random_state=1), x_train, y_train)
        #
        # save_model(model_1, path + 'model/' + data_set, 'neural_net_model_1')
        # save_model(model_2, path + 'model/' + data_set, 'neural_net_model_2')
        # save_model(model_3, path + 'model/' + data_set, 'neural_net_model_3')
        # save_model(model_4, path + 'model/' + data_set, 'neural_net_model_4')

    # else:
    #     print('Training Neural Network...')
    #     model_1 = MLPClassifier(solver='sgd', validation_fraction=0.0, alpha=1e-3, hidden_layer_sizes=(20, 5), random_state=1)
    #     model_2 = MLPClassifier(solver='sgd', validation_fraction=0.0, alpha=1e-3, hidden_layer_sizes=(50, 5), random_state=1)
    #     model_3 = MLPClassifier(solver='sgd', validation_fraction=0.0, alpha=1e-3, hidden_layer_sizes=(100, 5), random_state=1)
    #     model_4 = MLPClassifier(solver='sgd', validation_fraction=0.0, alpha=1e-3, hidden_layer_sizes=(500, 5), random_state=1)
    #
    #     plt = multiple_learning_curves_plot(
    #         [model_1, model_2, model_3, model_4],
    #         x_train, y_train,
    #         ["r", "y", "b", "m"],
    #         ['HLS = 20 x 5', 'HLS = 50 x 5', 'HLS = 100 x 5', 'HLS = 500 x 5']
    #     )
    #
    #     plt.title("Neural Network with Varying Hidden Layer Size (HLS) \n Learning Curves")
    #     plt.xlabel("Training examples")
    #     plt.ylabel("F1 Score")
    #     plt.grid()
    #
    #     plt.legend(loc="best")
    #     # plt.show()
    #     save_figure(plt, path + "plot/" + data_set, 'neural_net_learning_curves.png')
    #     print("done")


if __name__ == "__main__":
    # train_neural_net('../', False)
    train_neural_net_with_loan_data('../', "False")
