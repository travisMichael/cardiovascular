# from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from utils import save_model, load_data
from visualization_utils import multiple_learning_curves_plot
from utils import save_figure, train_and_time


def train_k_NN(path, with_plots):
    data_set = 'cardio'
    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')

    if not with_plots:
        model_1 = train_and_time(KNeighborsClassifier(n_neighbors=25), x_train, y_train)
        model_2 = train_and_time(KNeighborsClassifier(n_neighbors=150), x_train, y_train)
        model_3 = train_and_time(KNeighborsClassifier(n_neighbors=225), x_train, y_train)
        model_4 = train_and_time(KNeighborsClassifier(n_neighbors=300), x_train, y_train)

        save_model(model_1, path + 'model/' + data_set, 'kNN_model_1')
        save_model(model_2, path + 'model/' + data_set, 'kNN_model_2')
        save_model(model_3, path + 'model/' + data_set, 'kNN_model_3')
        save_model(model_4, path + 'model/' + data_set, 'kNN_model_4')

    else:
        print('Training kNN...')

        model_1 = KNeighborsClassifier(n_neighbors=25)
        model_2 = KNeighborsClassifier(n_neighbors=150)
        model_3 = KNeighborsClassifier(n_neighbors=225)
        model_4 = KNeighborsClassifier(n_neighbors=300)
        plt = multiple_learning_curves_plot(
            [model_1, model_2, model_3, model_4],
            x_train, y_train,
            ["r", "y", "b", "m"],
            ['k = 25', 'k = 150', 'k = 225', 'k = 300']
        )

        plt.title("k Nearest Neighbor \n Learning Curves")
        plt.xlabel("Training examples")
        plt.ylabel("F1 Score")
        plt.grid()

        plt.legend(loc="best")
        # plt.show()
        save_figure(plt, path + "plot/" + data_set, 'kNN_learning_curves.png')


def train_k_NN_loan(path, with_plots):
    data_set = 'loan'
    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')

    if not with_plots:
        model_1 = train_and_time(KNeighborsClassifier(n_neighbors=5), x_train, y_train)
        model_2 = train_and_time(KNeighborsClassifier(n_neighbors=10), x_train, y_train)
        model_3 = train_and_time(KNeighborsClassifier(n_neighbors=25), x_train, y_train)
        model_4 = train_and_time(KNeighborsClassifier(n_neighbors=100), x_train, y_train)

        save_model(model_1, path + 'model/' + data_set, 'kNN_model_1')
        save_model(model_2, path + 'model/' + data_set, 'kNN_model_2')
        save_model(model_3, path + 'model/' + data_set, 'kNN_model_3')
        save_model(model_4, path + 'model/' + data_set, 'kNN_model_4')

    else:
        print('Training kNN...')

        model_1 = KNeighborsClassifier(n_neighbors=5)
        model_2 = KNeighborsClassifier(n_neighbors=10)
        model_3 = KNeighborsClassifier(n_neighbors=25)
        model_4 = KNeighborsClassifier(n_neighbors=100)
        plt = multiple_learning_curves_plot(
            [model_1, model_2, model_3, model_4],
            x_train, y_train,
            ["r", "y", "b", "m"],
            ['k = 5', 'k = 10', 'k = 25', 'k = 100']
        )

        plt.title("k Nearest Neighbor \n Learning Curves")
        plt.xlabel("Training examples")
        plt.ylabel("F1 Score")
        plt.grid()

        plt.legend(loc="best")
        # plt.show()
        save_figure(plt, path + "plot/" + data_set, 'kNN_learning_curves.png')


if __name__ == "__main__":
    # train_k_NN('../', False)
    train_k_NN_loan('../', False)
