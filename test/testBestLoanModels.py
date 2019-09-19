import pickle
from visualization_utils import multiple_precision_recall_curves
from utils import save_figure


def test_best_models_loan(X, y, path):
    data_set = 'loan'
    probabilit_list = []
    dtc = pickle.load(open(path + 'model/' + data_set + '/dtc_model_depth_2', 'rb'))
    probs = dtc.predict_proba(X)
    probs = probs[:, 1]
    probabilit_list.append(probs)

    dtc = pickle.load(open(path + 'model/' + data_set + '/boosted_dtc_model_nodes_1', 'rb'))
    probs = dtc.predict_proba(X)
    probs = probs[:, 1]
    probabilit_list.append(probs)

    dtc = pickle.load(open(path + 'model/' + data_set + '/neural_net_model_4', 'rb'))
    probs = dtc.predict_proba(X)
    probs = probs[:, 1]
    probabilit_list.append(probs)

    # dtc = pickle.load(open(path + 'model/' + data_set + '/svm_model_1', 'rb'))
    # probs = dtc.predict_proba(X)
    # probs = probs[:, 1]
    # probabilit_list.append(probs)

    dtc = pickle.load(open(path + 'model/' + data_set + '/kNN_model_4', 'rb'))
    probs = dtc.predict_proba(X)
    probs = probs[:, 1]
    probabilit_list.append(probs)

    # with svm
    # color_list = ['r', 'b', 'm', 'y', 'g']
    # label_list = ['model = decision tree', 'model = boosted decision tree', 'model = neural network', 'model = SVM', 'model = kNN']

    # without svm
    color_list = ['r', 'b', 'm', 'g']
    label_list = ['model = decision tree', 'model = boosted decision tree', 'model = neural network', 'model = kNN']

    plt = multiple_precision_recall_curves(y, probabilit_list, color_list, label_list)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve of Best Model for each Algorithm')
    plt.legend(loc="best")

    # plt.show()
    save_figure(plt, path + "plot/" + data_set, 'best_models_pr_curve.png')
