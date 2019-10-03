# Resources
# https://buildmedia.readthedocs.org/media/pdf/mlrose/stable/mlrose.pdf
from utils import load_data
from mlrosevariation import neural, decay
from sklearn.metrics import log_loss, accuracy_score
import numpy as np
import time
# random_hill_climb
# simulated_annealing
# genetic_alg


def train_neural_net_with_loan_data(path, with_plots):
    data_set = 'loan'
    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')
    X_test, y_test = load_data(path + 'data/' + data_set + '/test/')

    if with_plots == "False":

        start_time = time.time()
        nn_ga = neural.NeuralNetwork(hidden_nodes = [40], activation = 'sigmoid',
                                         algorithm = 'genetic_alg', max_iters = 1000, pop_size=150, mutation_prob=0.999,
                                         bias = True, is_classifier = True, learning_rate = 0.001, curve=True,
                                         early_stopping = True, clip_max = 1, max_attempts =50, random_state = 3)
        f= open(path + "optimization/ga_stats.txt","w+")
        nn_ga.fit(x_train, y_train)
        np.save(path + "optimization/ga_fitness_curve", nn_ga.fitness_curve)
        end_time = time.time() - start_time
        f.write("SGD algorithm training time: " + str(end_time))
        print("Genetic algorithm training time: ", end_time)
        y_pred = nn_ga.predict_proba(X_test)
        test_loss = log_loss(y_test, y_pred)
        y_pred = nn_ga.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)

        y_pred = nn_ga.predict_proba(x_train)
        training_loss = log_loss(y_train, y_pred)
        y_pred = nn_ga.predict(x_train)
        training_accuracy = accuracy_score(y_train, y_pred)
        print("Loss: ", training_loss, test_loss, training_accuracy, test_accuracy)
        f.write("Loss: " + str(training_loss) + " " + str(test_loss) + " " + str(training_accuracy) + " " + str(test_accuracy))
        f.close()

        # -------------------------------------------------------------------------------------------------------------


        start_time = time.time()
        nn_sgd = neural.NeuralNetwork(hidden_nodes = [40], activation = 'sigmoid', curve=True,
                                         algorithm = 'gradient_descent', max_iters = 5000, pop_size=100, mutation_prob=0.3,
                                         bias = True, is_classifier = True, learning_rate = 0.00001,
                                         early_stopping = True, clip_max = 1, max_attempts =20, random_state = 3)
        f= open(path + "optimization/sgd_stats.txt","w+")
        nn_sgd.fit(x_train, y_train)
        np.save(path + "optimization/sgd_fitness_curve", nn_sgd.fitness_curve)
        end_time = time.time() - start_time
        f.write("SGD algorithm training time: " + str(end_time))
        print("SGD algorithm training time: ", end_time)
        y_pred = nn_sgd.predict_proba(X_test)
        test_loss = log_loss(y_test, y_pred)
        y_pred = nn_sgd.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)

        y_pred = nn_sgd.predict_proba(x_train)
        training_loss = log_loss(y_train, y_pred)
        y_pred = nn_sgd.predict(x_train)
        training_accuracy = accuracy_score(y_train, y_pred)
        print("Loss: ", training_loss, test_loss, training_accuracy, test_accuracy)
        f.write("Loss: " + str(training_loss) + " " + str(test_loss) + " " + str(training_accuracy) + " " + str(test_accuracy))
        f.close()

        decay_object = decay.ExpDecay(init_temp=.00001, exp_const=0.0005, min_temp=0.00000001)
        start_time = time.time()
        nn_sa = neural.NeuralNetwork(hidden_nodes = [40], activation ='sigmoid',
                                         algorithm = 'simulated_annealing', max_iters = 10000, schedule=decay_object,
                                         bias = True, is_classifier = True, learning_rate = 0.5, curve=True,
                                         early_stopping = True, clip_max = 1, max_attempts =15, random_state = 3)
        f= open(path + "optimization/sa_stats.txt","w+")
        nn_sa.fit(x_train, y_train)
        np.save(path + "optimization/sa_fitness_curve", nn_sa.fitness_curve)
        end_time = time.time() - start_time
        f.write("SA algorithm training time: " + str(end_time))
        print("SA algorithm training time: ", end_time)
        y_pred = nn_sa.predict_proba(X_test)
        test_loss = log_loss(y_test, y_pred)
        y_pred = nn_sa.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)

        y_pred = nn_sa.predict_proba(x_train)
        training_loss = log_loss(y_train, y_pred)
        y_pred = nn_sa.predict(x_train)
        training_accuracy = accuracy_score(y_train, y_pred)
        print("Loss: ", training_loss, test_loss, training_accuracy, test_accuracy)
        f.write("Loss: " + str(training_loss) + " " + str(test_loss) + " " + str(training_accuracy) + " " + str(test_accuracy))
        f.close()



        start_time = time.time()
        nn_rhc = neural.NeuralNetwork(hidden_nodes = [40], activation ='sigmoid',
                                         algorithm = 'random_hill_climb', max_iters = 5000,
                                         bias = True, is_classifier = True, learning_rate = 0.5, curve=True,
                                         early_stopping = True, clip_max = 1, max_attempts =15, random_state = 3)
        f= open(path + "optimization/rhc_stats.txt","w+")
        nn_rhc.fit(x_train, y_train)
        np.save(path + "optimization/rhc_fitness_curve", nn_rhc.fitness_curve)
        end_time = time.time() - start_time
        f.write("RHC algorithm training time: " + str(end_time))
        print("RHC algorithm training time: ", end_time)
        y_pred = nn_rhc.predict_proba(X_test)
        test_loss = log_loss(y_test, y_pred)
        y_pred = nn_rhc.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)

        y_pred = nn_rhc.predict_proba(x_train)
        training_loss = log_loss(y_train, y_pred)
        y_pred = nn_rhc.predict(x_train)
        training_accuracy = accuracy_score(y_train, y_pred)
        print("Loss: ", training_loss, test_loss, training_accuracy, test_accuracy)
        f.write("Loss: " + str(training_loss) + " " + str(test_loss) + " " + str(training_accuracy) + " " + str(test_accuracy))
        f.close()


if __name__ == "__main__":
    # train_neural_net('../', False)
    train_neural_net_with_loan_data('../', "False")
