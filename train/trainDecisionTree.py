from visualization_utils import multiple_learning_curves_plot
from sklearn import tree
from utils import save_model, save_figure, load_data, train_and_time


def train_dtc(path, with_plots):
    data_set = "cardio"
    print("Training Decision Tree Classifier...")

    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')

    if with_plots == "False":
        print("training 1")
        model_nodes_1 = train_and_time(tree.DecisionTreeClassifier(max_depth=5), x_train, y_train)
        model_nodes_2 = train_and_time(tree.DecisionTreeClassifier(max_depth=10), x_train, y_train)
        model_nodes_3 = train_and_time(tree.DecisionTreeClassifier(max_depth=15), x_train, y_train)
        model_nodes_4 = train_and_time(tree.DecisionTreeClassifier(max_depth=20), x_train, y_train)
        model_nodes_5 = train_and_time(tree.DecisionTreeClassifier(), x_train, y_train)

        save_model(model_nodes_1, path + "model/" + data_set, 'dtc_model_nodes_1')
        save_model(model_nodes_2, path + "model/" + data_set, 'dtc_model_nodes_2')
        save_model(model_nodes_3, path + "model/" + data_set, 'dtc_model_nodes_3')
        save_model(model_nodes_4, path + "model/" + data_set, 'dtc_model_nodes_4')
        save_model(model_nodes_5, path + "model/" + data_set, 'dtc_none')

        model_leaf_nodes_1 = train_and_time(tree.DecisionTreeClassifier(max_leaf_nodes=20), x_train, y_train)
        model_leaf_nodes_2 = train_and_time(tree.DecisionTreeClassifier(max_leaf_nodes=100), x_train, y_train)
        model_leaf_nodes_3 = train_and_time(tree.DecisionTreeClassifier(max_leaf_nodes=1000), x_train, y_train)
        model_leaf_nodes_4 = train_and_time(tree.DecisionTreeClassifier(max_leaf_nodes=2000), x_train, y_train)

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
    data_set = "loan"
    print("Training Decision Tree Classifier...")

    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')

    if with_plots == "False":
        model_nodes_1 = train_and_time(tree.DecisionTreeClassifier(max_depth=4), x_train, y_train)
        model_nodes_2 = train_and_time(tree.DecisionTreeClassifier(max_depth=8), x_train, y_train)
        model_nodes_3 = train_and_time(tree.DecisionTreeClassifier(max_depth=15), x_train, y_train)
        model_nodes_4 = train_and_time(tree.DecisionTreeClassifier(max_depth=30), x_train, y_train)
        model_nodes_5 = train_and_time(tree.DecisionTreeClassifier(), x_train, y_train)

        save_model(model_nodes_1, path + "model/" + data_set, 'dtc_model_depth_1')
        save_model(model_nodes_2, path + "model/" + data_set, 'dtc_model_depth_2')
        save_model(model_nodes_3, path + "model/" + data_set, 'dtc_model_depth_3')
        save_model(model_nodes_4, path + "model/" + data_set, 'dtc_model_depth_4')
        save_model(model_nodes_5, path + "model/" + data_set, 'dtc_none')

        model_leaf_nodes_1 = train_and_time(tree.DecisionTreeClassifier(max_leaf_nodes=5), x_train, y_train)
        model_leaf_nodes_2 = train_and_time(tree.DecisionTreeClassifier(max_leaf_nodes=20), x_train, y_train)
        model_leaf_nodes_3 = train_and_time(tree.DecisionTreeClassifier(max_leaf_nodes=100), x_train, y_train)
        model_leaf_nodes_4 = train_and_time(tree.DecisionTreeClassifier(max_leaf_nodes=300), x_train, y_train)

        save_model(model_leaf_nodes_1, path + "model/" + data_set, 'dtc_model_leaf_nodes_1')
        save_model(model_leaf_nodes_2, path + "model/" + data_set, 'dtc_model_leaf_nodes_2')
        save_model(model_leaf_nodes_3, path + "model/" + data_set, 'dtc_model_leaf_nodes_3')
        save_model(model_leaf_nodes_4, path + "model/" + data_set, 'dtc_model_leaf_nodes_4')

    else:
        # model_1 = tree.DecisionTreeClassifier(max_leaf_nodes=5)
        # model_2 = tree.DecisionTreeClassifier(max_leaf_nodes=20)
        # model_3 = tree.DecisionTreeClassifier(max_leaf_nodes=100)
        # model_4 = tree.DecisionTreeClassifier(max_leaf_nodes=300)
        # model_5 = tree.DecisionTreeClassifier()
        #
        # plt = multiple_learning_curves_plot(
        #     [model_1, model_2, model_3, model_4, model_5],
        #     x_train, y_train,
        #     ["r", "y", "g", "m", "b"],
        #     ["MLN = 5", "MLN = 20", "MLN = 100", "MLN = 300", "MLN = None"]
        # )
        #
        # plt.title("Decision Tree Learning Curves \n With Max Leaf Nodes (MLN)")
        # plt.xlabel("Training examples")
        # plt.ylabel("F1 Score")
        # plt.grid()
        # plt.legend(loc="best")
        # save_figure(plt, path + "plot/" + data_set, 'dtc_mln_learning_curve.png')

        # ---------------------------------------------------------------------

        model_1 = tree.DecisionTreeClassifier(max_depth=4)
        model_2 = tree.DecisionTreeClassifier(max_depth=8)
        model_3 = tree.DecisionTreeClassifier(max_depth=15)
        model_4 = tree.DecisionTreeClassifier(max_depth=30)
        model_5 = tree.DecisionTreeClassifier()
        plt = multiple_learning_curves_plot(
            [model_1, model_2, model_3, model_4, model_5],
            x_train, y_train,
            ["r", "y", "g", "m", "b"],
            ["MD = 4", "MD = 8", "MD = 15", "MD = 30", "MD = None"]
        )
        plt.title("Decision Tree Learning Curves \n With Max Depth (MD)")
        plt.xlabel("Training examples")
        plt.ylabel("F1 Score")
        plt.grid()
        plt.legend(loc="best")
        save_figure(plt, path + "plot/" + data_set, 'dtc_md_learning_curve.png')


if __name__ == "__main__":
    # train_dtc('../', False)
    train_dtc_loan('../', False)
