# Resources
# https://scikit-learn.org/stable/modules/neural_networks_supervised.html
# https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mlp_training_curves.html
from sklearn.neural_network import MLPClassifier
import numpy as np
import pickle
from utils import save_model


def train_neural_net(dataset, path):
    print('Training Neural Network...')
    model = MLPClassifier(solver='sgd', validation_fraction=0.0, alpha=1e-4, hidden_layer_sizes=(10, 10), random_state=1)

    x_train_file = open(path + 'data/train/x', 'rb')
    y_train_file = open(path + 'data/train/y', 'rb')
    x_train = pickle.load(x_train_file)
    y_train = pickle.load(y_train_file)

    model.fit(x_train, y_train)

    save_model(model, dataset, 'best_neural_net_model')

    x_train_file.close()
    y_train_file.close()
    print("done")


if __name__ == "__main__":
    print('hello')
    train_neural_net('cardio', '../')
