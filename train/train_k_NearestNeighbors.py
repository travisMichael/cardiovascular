# from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from utils import save_model, load_data
from visualization_utils import multiple_learning_curves_plot

def train_k_NN(data_set, path):
    print('Training kNN...')
    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')

    # model = NearestNeighbors(n_neighbors=201, algorithm='ball_tree').fit(x_train)
    # distances, indices = model.kneighbors(x_test)
    # model = KNeighborsClassifier(n_neighbors=200).fit(x_train, y_train)
    # results = model.predict(x_test)

    model_2 = KNeighborsClassifier(n_neighbors=75)
    model_3 = KNeighborsClassifier(n_neighbors=150)
    model_4 = KNeighborsClassifier(n_neighbors=225)
    model_5 = KNeighborsClassifier(n_neighbors=300)
    model_6 = KNeighborsClassifier(n_neighbors=50).fit(x_train, y_train)
    # plt = multiple_learning_curves_plot(
    #     [model_2],
    #     x_train, y_train,
    #     ["r", "y", "b", "g", "m"],
    #     ["Max depth = 3", "Max depth = 4", "Max depth = 5", "Max depth = 6", "Max depth = 7"]
    # )
    #
    # plt.title("Title")
    # plt.xlabel("Training examples")
    # plt.ylabel("Score")
    # plt.grid()
    #
    # plt.legend(loc="best")
    # plt.show()

    save_model(model_6, data_set, 'best_kNN_model')
    print("done")


if __name__ == "__main__":
    train_k_NN('cardio', '../')
