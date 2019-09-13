import pickle
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
from utils import load_data


def test_model(model_to_test, path, data_set):
    X, y = load_data(path + 'data/' + data_set + '/train/')

    if model_to_test == 'all':
        test_boosted(X, y)
        test_decision_tree(X, y)
        test_kNN(X, y)
        test_neural_net(X, y)
        test_svm(X, y)
    elif model_to_test == 'kNN':
        test_kNN(X, y, path, data_set)
    elif model_to_test == 'boosted':
        test_neural_net(X, y)
    elif model_to_test == 'dtc':
        test_neural_net(X, y)
    elif model_to_test == 'neural_net':
        test_neural_net(X, y)
    elif model_to_test == 'svm':
        test_neural_net(X, y)


def test_kNN(X, y, path, data_set):
    kNN = pickle.load(open(path + 'model/' + data_set + '/best_kNN_model', 'rb'))
    kNN_results = kNN.predict(X)
    kNN_average_precision = average_precision_score(y, kNN_results)
    print("k NN Results: ", kNN_average_precision)


def test_boosted(X, y):
    boosted_model = pickle.load(open('../model/best_boosted_model', 'rb'))
    boosted_model_results = boosted_model.predict(X)
    boosted_model_average_precision = average_precision_score(y, boosted_model_results)
    print("Boosted Decision Tree Results: ", boosted_model_average_precision)


def test_decision_tree(X, y):
    dtc = pickle.load(open('../model/best_decision_tree_model', 'rb'))
    dtc_results = dtc.predict(X)
    dtc_average_precision = average_precision_score(y, dtc_results)
    print("Decision Tree Results: ", dtc_average_precision)


def test_neural_net(X, y):
    neural_net = pickle.load(open('../model/loan/best_neural_net_model', 'rb'))
    neural_net_results = neural_net.predict(X)
    neural_net_average_precision = average_precision_score(y, neural_net_results)
    print("Neural Net Results: ", neural_net_average_precision)
    # neural_net_results
    # cm = confusion_matrix(y_test, neural_net_results)
    # cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


def test_svm(X, y):
    svn = pickle.load(open('../model/best_SVN_model', 'rb'))
    svn_results = svn.predict(X)
    svn_average_precision = average_precision_score(y, svn_results)
    print("SVN Results: ", svn_average_precision)


if __name__ == "__main__":
    test_model('kNN', '../', 'loan')

