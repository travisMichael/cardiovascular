import pickle
from sklearn.metrics import average_precision_score
from utils import load_data
from test.testDTC import test_decision_tree, test_decision_tree_max_leaf
from test.testNeuralNet import test_neural_net_cardio
from test.testKNN import test_kNN


def test_model(model_to_test, path, data_set):
    X, y = load_data(path + 'data/' + data_set + '/test/')

    if model_to_test == 'all':
        test_boosted(X, y)
        test_decision_tree(X, y, path, data_set)
        test_kNN(X, y)
        test_neural_net_cardio(X, y, path)
        test_svm(X, y)
    elif model_to_test == 'kNN':
        test_kNN(X, y, path)
    elif model_to_test == 'boosted':
        test_boosted(X, y)
    elif model_to_test == 'dtc':
        test_decision_tree(X, y, path, data_set)
        test_decision_tree_max_leaf(X, y, path, data_set)
    elif model_to_test == 'neural_net':
        test_neural_net_cardio(X, y, path)
    elif model_to_test == 'svm':
        test_svm(X, y)


def test_boosted(X, y):
    boosted_model = pickle.load(open('../model/best_boosted_model', 'rb'))
    boosted_model_results = boosted_model.predict(X)
    boosted_model_average_precision = average_precision_score(y, boosted_model_results)
    print("Boosted Decision Tree Results: ", boosted_model_average_precision)


def test_svm(X, y):
    svn = pickle.load(open('../model/best_SVN_model', 'rb'))
    svn_results = svn.predict(X)
    svn_average_precision = average_precision_score(y, svn_results)
    print("SVN Results: ", svn_average_precision)


if __name__ == "__main__":
    test_model('kNN', '../', 'cardio')

