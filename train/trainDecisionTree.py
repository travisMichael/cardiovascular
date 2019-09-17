from visualization_utils import multiple_learning_curves_plot, multiple_precision_recall_curves
from sklearn import tree
from utils import save_model, save_figure, load_data


def train_dtc(path, with_plots):
    data_set = "cardio"
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
        # model_1 = tree.DecisionTreeClassifier(max_leaf_nodes=20)
        # model_2 = tree.DecisionTreeClassifier(max_leaf_nodes=100)
        # model_3 = tree.DecisionTreeClassifier(max_leaf_nodes=1000)
        # model_4 = tree.DecisionTreeClassifier(max_leaf_nodes=2000)
        # model_5 = tree.DecisionTreeClassifier()
        #
        # plt = multiple_learning_curves_plot(
        #     [model_1, model_2, model_3, model_4, model_5],
        #     x_train, y_train,
        #     ["r", "y", "g", "m", "b"],
        #     ["MLN = 20", "MLN = 100", "MLN = 1000", "MLN = 2000", "MLN = None"]
        # )
        #
        # plt.title("Decision Tree Learning Curves \n With Max Leaf Nodes (MLN)")
        # plt.xlabel("Training examples")
        # plt.ylabel("F1 Score")
        # plt.grid()
        # plt.legend(loc="best")
        # save_figure(plt, path + "plot/" + data_set, 'dtc_mln_plots.png')

        # ---------------------------------------------------------------------

        model_1 = tree.DecisionTreeClassifier(max_depth=5)
        model_2 = tree.DecisionTreeClassifier(max_depth=10)
        model_3 = tree.DecisionTreeClassifier(max_depth=15)
        model_4 = tree.DecisionTreeClassifier(max_depth=20)
        model_5 = tree.DecisionTreeClassifier()
        plt = multiple_learning_curves_plot(
            [model_1, model_2, model_3, model_4, model_5],
            x_train, y_train,
            ["r", "y", "g", "m", "b"],
            ["MD = 5", "MD = 10", "MD = 15", "MD = 20", "MD = None"]
        )
        plt.title("Decision Tree Learning Curves \n With Max Depth (MD)")
        plt.xlabel("Training examples")
        plt.ylabel("F1 Score")
        plt.grid()
        plt.legend(loc="best")
        save_figure(plt, path + "plot/" + data_set, 'dtc_md_plots.png')


def train_dtc_loan(path, with_plots):
    data_set = "cardio"
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
        model_1 = tree.DecisionTreeClassifier(max_leaf_nodes=20)
        model_2 = tree.DecisionTreeClassifier(max_leaf_nodes=100)
        model_3 = tree.DecisionTreeClassifier(max_leaf_nodes=1000)
        model_4 = tree.DecisionTreeClassifier(max_leaf_nodes=2000)
        model_5 = tree.DecisionTreeClassifier()

        plt = multiple_learning_curves_plot(
            [model_1, model_2, model_3, model_4, model_5],
            x_train, y_train,
            ["r", "y", "g", "m", "b"],
            ["MLN = 20", "MLN = 100", "MLN = 1000", "MLN = 2000", "MLN = None"]
        )

        plt.title("Decision Tree Learning Curves \n With Max Leaf Nodes (MLN)")
        plt.xlabel("Training examples")
        plt.ylabel("F1 Score")
        plt.grid()
        plt.legend(loc="best")
        save_figure(plt, path + "plot/" + data_set, 'dtc_mln_plots.png')

        # ---------------------------------------------------------------------


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

if __name__ == "__main__":
    train_dtc('../', True)
