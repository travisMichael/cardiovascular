from visualization_utils import multiple_learning_curves_plot
from sklearn import tree
from utils import save_model, save_figure, load_data


def train_dtc(data_set, path):
    print("Training Decision Tree Classifier...")

    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')

    # model_2 = tree.DecisionTreeClassifier(max_depth=3)
    # model_3 = tree.DecisionTreeClassifier(max_depth=4)
    # model_4 = tree.DecisionTreeClassifier(max_depth=5)
    # model_5 = tree.DecisionTreeClassifier(max_depth=6)
    # model_6 = tree.DecisionTreeClassifier(max_depth=7)
    # plt = multiple_learning_curves_plot(
    #     [model_2, model_3, model_4, model_5, model_6],
    #     x_train, y_train,
    #     ["r", "y", "b", "g", "m"],
    #     ["Max depth = 3", "Max depth = 4", "Max depth = 5", "Max depth = 6", "Max depth = 7"]
    # )

    model_2 = tree.DecisionTreeClassifier(max_depth=5, max_leaf_nodes=60)
    model_3 = tree.DecisionTreeClassifier(max_depth=5, max_leaf_nodes=50)
    model_4 = tree.DecisionTreeClassifier(max_depth=5, max_leaf_nodes=40)
    model_5 = tree.DecisionTreeClassifier(max_depth=5, max_leaf_nodes=30)
    model_6 = tree.DecisionTreeClassifier(max_depth=5, max_leaf_nodes=20)
    # model_5.fit(x_train, y_train)
    plt = multiple_learning_curves_plot(
        [model_2],
        x_train, y_train,
        ["r", "y", "b", "g", "m"],
        ["Max depth = 3", "Max depth = 4", "Max depth = 5", "Max depth = 6", "Max depth = 7"]
    )

    # plt.title("Title")
    # plt.xlabel("Training examples")
    # plt.ylabel("Score")
    # plt.grid()
    #
    # plt.legend(loc="best")
    # plt.show()

    save_model(model_5, path + "model/" + data_set, 'best_decision_tree_model')
    save_figure(plt, path + "plot/" + data_set, 'dtc_plots.png')
    print("done")


if __name__ == "__main__":
    train_dtc('loan', '../')
