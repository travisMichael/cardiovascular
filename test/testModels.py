import pickle
from sklearn.metrics import average_precision_score
from utils import load_data
from visualization_utils import multiple_precision_recall_curves
from test.testDTC import test_decision_tree, test_decision_tree_max_leaf


def test_model(model_to_test, path, data_set):
    X, y = load_data(path + 'data/' + data_set + '/test/')

    if model_to_test == 'all':
        test_boosted(X, y)
        test_decision_tree(X, y)
        test_kNN(X, y)
        test_neural_net(X, y)
        test_svm(X, y)
    elif model_to_test == 'kNN':
        test_kNN(X, y, path, data_set)
    elif model_to_test == 'boosted':
        test_neural_net(X, y, data_set)
    elif model_to_test == 'dtc':
        test_decision_tree(X, y, path, data_set)
        test_decision_tree_max_leaf(X, y, path, data_set)
    elif model_to_test == 'neural_net':
        test_neural_net(X, y, data_set)
    elif model_to_test == 'svm':
        test_neural_net(X, y, data_set)


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


def test_neural_net(X, y, data_set):
    probabilit_list = []
    dtc = pickle.load(open('../model/' + data_set + '/dtc_model_nodes_1', 'rb'))
    probs = dtc.predict_proba(X)
    probs = probs[:, 1]
    probabilit_list.append(probs)

    dtc = pickle.load(open('../model/' + data_set + '/best_neural_net_model', 'rb'))
    probs = dtc.predict_proba(X)
    probs = probs[:, 1]
    probabilit_list.append(probs)

    color_list = ['r', 'b']
    label_list = ['one', 'two']

    plt = multiple_precision_recall_curves(y, probabilit_list, color_list, label_list)

    plt.show()
    dtc_average_precision = average_precision_score(y, probabilit_list[0])
    print("Decision Tree Results: ", dtc_average_precision)
    dtc_average_precision = average_precision_score(y, dtc.predict(X))
    print("Decision Tree Results: ", dtc_average_precision)


def test_svm(X, y):
    svn = pickle.load(open('../model/best_SVN_model', 'rb'))
    svn_results = svn.predict(X)
    svn_average_precision = average_precision_score(y, svn_results)
    print("SVN Results: ", svn_average_precision)


if __name__ == "__main__":
    test_model('dtc', '../', 'cardio')

