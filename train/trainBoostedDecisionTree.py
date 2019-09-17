# https://statinfer.com/204-3-10-pruning-a-decision-tree-in-python/
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from utils import save_model, load_data


def train_boosted_dtc(data_set, path, with_plots):

    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')

    if not with_plots:
        print("training 1")
        model_nodes_1 = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=5)).fit(x_train, y_train)
        print("training 2")
        model_nodes_2 = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=10)).fit(x_train, y_train)
        print("training 3")
        model_nodes_3 = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=15)).fit(x_train, y_train)
        print("training 4")
        model_nodes_4 = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=20)).fit(x_train, y_train)
        model_nodes_5 = AdaBoostClassifier(tree.DecisionTreeClassifier()).fit(x_train, y_train)

        save_model(model_nodes_1, path + "model/" + data_set, 'boosted_dtc_model_nodes_1')
        save_model(model_nodes_2, path + "model/" + data_set, 'boosted_dtc_model_nodes_2')
        save_model(model_nodes_3, path + "model/" + data_set, 'boosted_dtc_model_nodes_3')
        save_model(model_nodes_4, path + "model/" + data_set, 'boosted_dtc_model_nodes_4')
        save_model(model_nodes_5, path + "model/" + data_set, 'boosted_dtc_none')

    else:
        print('Training boosted dtc...')
        # model = tree.DecisionTreeClassifier() max_depth=4
        model = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=3), )

        model.fit(x_train, y_train)
        # model.partial_fit(x_train, y_train, classes=N_CLASSES)
        # result = model.predict(x_test)

        save_model(model, path + 'model/' + data_set, 'best_boosted_dtc_model')

        print("done")


if __name__ == "__main__":
    train_boosted_dtc('cardio', '../', False)
