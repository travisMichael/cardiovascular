# https://statinfer.com/204-3-10-pruning-a-decision-tree-in-python/
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from utils import save_model, load_data


def train_boosted_dtc(data_set, path):
    print('Training boosted dtc...')
    # model = tree.DecisionTreeClassifier() max_depth=4
    model = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=3), )

    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')

    model.fit(x_train, y_train)
    # model.partial_fit(x_train, y_train, classes=N_CLASSES)
    # result = model.predict(x_test)

    save_model(model, path + 'model/' + data_set, 'best_boosted_dtc_model')

    print("done")


if __name__ == "__main__":
    train_boosted_dtc('cardio', '../')
