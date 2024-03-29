import pickle
from visualization_utils import multiple_precision_recall_curves
from utils import save_figure, calculate_f1_score


def test_boosted_decision_tree(X, y, path):
    data_set = 'cardio'
    probabilit_list = []
    dtc = pickle.load(open(path + 'model/' + data_set + '/boosted_dtc_model_nodes_1', 'rb'))
    probs = dtc.predict_proba(X)
    probs = probs[:, 1]
    probabilit_list.append(probs)
    calculate_f1_score(dtc, X, y)

    dtc = pickle.load(open(path + 'model/' + data_set + '/boosted_dtc_model_nodes_2', 'rb'))
    probs = dtc.predict_proba(X)
    probs = probs[:, 1]
    probabilit_list.append(probs)
    calculate_f1_score(dtc, X, y)

    dtc = pickle.load(open(path + 'model/' + data_set + '/boosted_dtc_model_nodes_3', 'rb'))
    probs = dtc.predict_proba(X)
    probs = probs[:, 1]
    probabilit_list.append(probs)
    calculate_f1_score(dtc, X, y)

    dtc = pickle.load(open(path + 'model/' + data_set + '/boosted_dtc_model_nodes_4', 'rb'))
    probs = dtc.predict_proba(X)
    probs = probs[:, 1]
    probabilit_list.append(probs)
    calculate_f1_score(dtc, X, y)

    dtc = pickle.load(open(path + 'model/' + data_set + '/boosted_dtc_none', 'rb'))
    probs = dtc.predict_proba(X)
    probs = probs[:, 1]
    probabilit_list.append(probs)
    calculate_f1_score(dtc, X, y)

    color_list = ['r', 'y', 'g', 'm', 'b']
    label_list = ['MD = 5', 'MD = 10', 'MD = 15', 'MD = 20', 'MD = None']

    plt = multiple_precision_recall_curves(y, probabilit_list, color_list, label_list)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.5, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Boosted Decision Tree Classifier with Max Depth Pruning (MD) \n Precision-Recall Curve ')
    plt.legend(loc="best")

    save_figure(plt, path + "plot/" + data_set, 'boosted_dtc_max_depth_plots.png')


def test_boosted_decision_tree_loan(X, y, path):
    data_set = 'loan'
    probabilit_list = []
    dtc = pickle.load(open(path + 'model/' + data_set + '/boosted_dtc_model_nodes_1', 'rb'))
    probs = dtc.predict_proba(X)
    probs = probs[:, 1]
    probabilit_list.append(probs)
    calculate_f1_score(dtc, X, y)

    dtc = pickle.load(open(path + 'model/' + data_set + '/boosted_dtc_model_nodes_2', 'rb'))
    probs = dtc.predict_proba(X)
    probs = probs[:, 1]
    probabilit_list.append(probs)
    calculate_f1_score(dtc, X, y)

    dtc = pickle.load(open(path + 'model/' + data_set + '/boosted_dtc_model_nodes_3', 'rb'))
    probs = dtc.predict_proba(X)
    probs = probs[:, 1]
    probabilit_list.append(probs)
    calculate_f1_score(dtc, X, y)

    dtc = pickle.load(open(path + 'model/' + data_set + '/boosted_dtc_model_nodes_4', 'rb'))
    probs = dtc.predict_proba(X)
    probs = probs[:, 1]
    probabilit_list.append(probs)
    calculate_f1_score(dtc, X, y)

    dtc = pickle.load(open(path + 'model/' + data_set + '/boosted_dtc_none', 'rb'))
    probs = dtc.predict_proba(X)
    probs = probs[:, 1]
    probabilit_list.append(probs)
    calculate_f1_score(dtc, X, y)

    color_list = ['r', 'y', 'g', 'm', 'b']
    label_list = ['MD = 4', 'MD = 8', 'MD = 15', 'MD = 30', 'MD = None']

    plt = multiple_precision_recall_curves(y, probabilit_list, color_list, label_list)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Boosted Decision Tree Classifier with Max Depth Pruning (MD) \n Precision-Recall Curve ')
    plt.legend(loc="best")

    save_figure(plt, path + "plot/" + data_set, 'boosted_dtc_max_depth_plots.png')
