# https://statinfer.com/204-3-10-pruning-a-decision-tree-in-python/
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
import pickle
from utils import save_model


def train_boosted_dtc(dataset, path):
    print('Training boosted dtc...')
    # model = tree.DecisionTreeClassifier() max_depth=4
    model = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=3), )

    x_train_file = open(path + 'data/train/x', 'rb')
    y_train_file = open(path + 'data/train/y', 'rb')
    x_train = pickle.load(x_train_file)
    y_train = pickle.load(y_train_file)

    model.fit(x_train, y_train)
    # model.partial_fit(x_train, y_train, classes=N_CLASSES)
    # result = model.predict(x_test)

    save_model(model, dataset, 'best_boosted_dtc_model')

    x_train_file.close()
    y_train_file.close()
    print("done")


if __name__ == "__main__":
    print('hello')
    train_boosted_dtc('cardio', '../')
