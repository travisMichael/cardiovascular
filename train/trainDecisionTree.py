from visualization_utils import multiple_learning_curves_plot, multiple_precision_recall_curves
from sklearn import tree
from utils import save_model, save_figure, load_data
from sklearn.metrics import average_precision_score, recall_score, precision_score
import numpy as np


def train_dtc(data_set, path, with_plots):
    print("Training Decision Tree Classifier...")

    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')

    if not with_plots:

        print("training 1")
        model_nodes_1 = tree.DecisionTreeClassifier(max_depth=5).fit(x_train, y_train)
        print("training 2")
        model_nodes_2 = tree.DecisionTreeClassifier(max_depth=10).fit(x_train, y_train)
        print("training 3")
        model_nodes_3 = tree.DecisionTreeClassifier(max_depth=15).fit(x_train, y_train)
        print("training 4")
        model_nodes_4 = tree.DecisionTreeClassifier(max_depth=20).fit(x_train, y_train)
        model_nodes_5 = tree.DecisionTreeClassifier().fit(x_train, y_train)

        save_model(model_nodes_1, path + "model/" + data_set, 'dtc_model_nodes_1')
        save_model(model_nodes_2, path + "model/" + data_set, 'dtc_model_nodes_2')
        save_model(model_nodes_3, path + "model/" + data_set, 'dtc_model_nodes_3')
        save_model(model_nodes_4, path + "model/" + data_set, 'dtc_model_nodes_4')
        save_model(model_nodes_5, path + "model/" + data_set, 'dtc_none')

        print("training 6")
        model_leaf_nodes_1 = tree.DecisionTreeClassifier(max_leaf_nodes=20).fit(x_train, y_train)
        print("training 7")
        model_leaf_nodes_2 = tree.DecisionTreeClassifier(max_leaf_nodes=100).fit(x_train, y_train)
        print("training 8")
        model_leaf_nodes_3 = tree.DecisionTreeClassifier(max_leaf_nodes=1000).fit(x_train, y_train)
        print("training 5")
        model_leaf_nodes_4 = tree.DecisionTreeClassifier(max_leaf_nodes=2000).fit(x_train, y_train)

        save_model(model_leaf_nodes_1, path + "model/" + data_set, 'dtc_model_leaf_nodes_1')
        save_model(model_leaf_nodes_2, path + "model/" + data_set, 'dtc_model_leaf_nodes_2')
        save_model(model_leaf_nodes_3, path + "model/" + data_set, 'dtc_model_leaf_nodes_3')
        save_model(model_leaf_nodes_4, path + "model/" + data_set, 'dtc_model_leaf_nodes_4')

    else:
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

        model_2 = tree.DecisionTreeClassifier(max_leaf_nodes=4000)
        model_3 = tree.DecisionTreeClassifier(max_leaf_nodes=50)
        model_4 = tree.DecisionTreeClassifier(max_leaf_nodes=40)
        model_5 = tree.DecisionTreeClassifier(max_leaf_nodes=30)
        model_6 = tree.DecisionTreeClassifier(max_leaf_nodes=20)
        # model_5.fit(x_train, y_train)
        plt = multiple_learning_curves_plot(
            [model_2],
            x_train, y_train,
            ["r", "y", "b", "g", "m"],
            ["Max depth = 3", "Max depth = 4", "Max depth = 5", "Max depth = 6", "Max depth = 7"]
        )

        plt.title("Decision Tree Learning Curves (with varying leaf nodes)")
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        plt.grid()

        # plt.legend(loc="best")
        plt.show()

        # save_figure(plt, path + "plot/" + data_set, 'dtc_plots.png')
        # print("done")
        save_model(model_2, path + "model/" + data_set, 'dtc_model_leaf_nodes_4')


if __name__ == "__main__":
    train_dtc('cardio', '../', False)
