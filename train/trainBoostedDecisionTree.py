# https://statinfer.com/204-3-10-pruning-a-decision-tree-in-python/
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from utils import save_model, load_data, save_figure, train_and_time
from visualization_utils import multiple_learning_curves_plot


def train_boosted_dtc(path, with_plots):
    data_set = 'cardio'
    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')

    if not with_plots:
        model_nodes_1 = train_and_time(AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=5)), x_train, y_train)
        model_nodes_2 = train_and_time(AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=10)), x_train, y_train)
        model_nodes_3 = train_and_time(AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=15)), x_train, y_train)
        model_nodes_4 = train_and_time(AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=20)), x_train, y_train)
        model_nodes_5 = train_and_time(AdaBoostClassifier(tree.DecisionTreeClassifier()), x_train, y_train)

        save_model(model_nodes_1, path + "model/" + data_set, 'boosted_dtc_model_nodes_1')
        save_model(model_nodes_2, path + "model/" + data_set, 'boosted_dtc_model_nodes_2')
        save_model(model_nodes_3, path + "model/" + data_set, 'boosted_dtc_model_nodes_3')
        save_model(model_nodes_4, path + "model/" + data_set, 'boosted_dtc_model_nodes_4')
        save_model(model_nodes_5, path + "model/" + data_set, 'boosted_dtc_none')

    else:
        print('Training boosted dtc...')
        model_1 = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=5))
        model_2 = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=10))
        model_3 = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=15))
        model_4 = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=20))
        model_5 = AdaBoostClassifier(tree.DecisionTreeClassifier())
        plt = multiple_learning_curves_plot(
            [model_1, model_2, model_3, model_4, model_5],
            x_train, y_train,
            ["r", "y", "g", "m", "b"],
            ["MD = 5", "MD = 10", "MD = 15", "MD = 20", "MD = None"]
        )
        plt.title("Boosted Decision Tree With Max Depth (MD) \n Pruning Learning Curves")
        plt.xlabel("Training examples")
        plt.ylabel("F1 Score")
        plt.grid()
        plt.legend(loc="best")
        save_figure(plt, path + "plot/" + data_set, 'boosted_dtc_md_learning_curves.png')


def train_boosted_dtc_loan(path, with_plots):
    data_set = 'loan'
    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')

    if not with_plots:
        model_nodes_1 = train_and_time(AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=4)), x_train, y_train)
        model_nodes_2 = train_and_time(AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=8)), x_train, y_train)
        model_nodes_3 = train_and_time(AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=15)), x_train, y_train)
        model_nodes_4 = train_and_time(AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=30)), x_train, y_train)
        model_nodes_5 = train_and_time(AdaBoostClassifier(tree.DecisionTreeClassifier()), x_train, y_train)

        save_model(model_nodes_1, path + "model/" + data_set, 'boosted_dtc_model_nodes_1')
        save_model(model_nodes_2, path + "model/" + data_set, 'boosted_dtc_model_nodes_2')
        save_model(model_nodes_3, path + "model/" + data_set, 'boosted_dtc_model_nodes_3')
        save_model(model_nodes_4, path + "model/" + data_set, 'boosted_dtc_model_nodes_4')
        save_model(model_nodes_5, path + "model/" + data_set, 'boosted_dtc_none')

    else:
        print('Training boosted dtc...')
        model_1 = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=4))
        model_2 = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=8))
        model_3 = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=15))
        model_4 = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=30))
        model_5 = AdaBoostClassifier(tree.DecisionTreeClassifier())
        plt = multiple_learning_curves_plot(
            [model_1, model_2, model_3, model_4, model_5],
            x_train, y_train,
            ["r", "y", "g", "m", "b"],
            ["MD = 4", "MD = 8", "MD = 15", "MD = 30", "MD = None"]
        )
        plt.title("Boosted Decision Tree With Max Depth (MD) \n Pruning Learning Curves")
        plt.xlabel("Training examples")
        plt.ylabel("F1 Score")
        plt.grid()
        plt.legend(loc="best")
        save_figure(plt, path + "plot/" + data_set, 'boosted_dtc_md_learning_curves.png')


if __name__ == "__main__":
    # train_boosted_dtc('../', False)
    train_boosted_dtc_loan('../', False)
