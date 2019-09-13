# Resources
# https://scikit-learn.org/stable/modules/neural_networks_supervised.html
# https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mlp_training_curves.html
from sklearn.neural_network import MLPClassifier
from visualization_utils import multiple_learning_curves_plot
from utils import save_model, load_data


def train_neural_net(data_set, path):
    print('Training Neural Network...')
    model = MLPClassifier(solver='sgd', validation_fraction=0.0, alpha=1e-4, hidden_layer_sizes=(10, 10), random_state=1)

    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')

    model.fit(x_train, y_train)

    # plt = multiple_learning_curves_plot(
    #     [model],
    #     x_train, y_train,
    #     ["r", "y", "b", "g", "m"],
    #     ["Max depth = 3", "Max depth = 4", "Max depth = 5", "Max depth = 6", "Max depth = 7"]
    # )

    # plt.title("Title")
    # plt.xlabel("Training examples")
    # plt.ylabel("Score")
    # plt.grid()
    #
    # plt.legend(loc="best")
    # plt.show()

    save_model(model, path + 'model/' + data_set, 'best_neural_net_model')
    print("done")


if __name__ == "__main__":
    train_neural_net('loan', '../')
