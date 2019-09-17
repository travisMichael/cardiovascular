from utils import load_data
from test.testDTC import test_decision_tree, test_decision_tree_max_leaf
from test.testNeuralNet import test_neural_net_cardio
from test.testKNN import test_kNN
from test.testBoostedDTC import test_boosted_decision_tree
from test.testSVM import test_svm


def test_model(model_to_test, path, data_set):
    X, y = load_data(path + 'data/' + data_set + '/test/')

    if model_to_test == 'all':
        test_boosted_decision_tree(X, y, path)
        test_decision_tree(X, y, path, data_set)
        test_kNN(X, y)
        test_neural_net_cardio(X, y, path)
        test_svm(X, y)
    elif model_to_test == 'kNN':
        test_kNN(X, y, path)
    elif model_to_test == 'boosted':
        test_boosted_decision_tree(X, y, path)
    elif model_to_test == 'dtc':
        test_decision_tree(X, y, path, data_set)
        test_decision_tree_max_leaf(X, y, path, data_set)
    elif model_to_test == 'neural_net':
        test_neural_net_cardio(X, y, path)
    elif model_to_test == 'svm':
        test_svm(X, y, path)


if __name__ == "__main__":
    test_model('svm', '../', 'cardio')

