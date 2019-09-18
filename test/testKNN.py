import pickle
from visualization_utils import multiple_precision_recall_curves
from utils import save_figure, calculate_f1_score


def test_kNN(X, y, path):
    data_set = 'cardio'

    print("Predicting 1")
    probabilit_list = []
    dtc = pickle.load(open('../model/' + data_set + '/kNN_model_1', 'rb'))
    probs = dtc.predict_proba(X)
    probs = probs[:, 1]
    probabilit_list.append(probs)
    calculate_f1_score(dtc, X, y)

    print("Predicting 2")
    dtc = pickle.load(open('../model/' + data_set + '/kNN_model_2', 'rb'))
    probs = dtc.predict_proba(X)
    probs = probs[:, 1]
    probabilit_list.append(probs)
    calculate_f1_score(dtc, X, y)

    print("Predicting 3")
    dtc = pickle.load(open('../model/' + data_set + '/kNN_model_3', 'rb'))
    probs = dtc.predict_proba(X)
    probs = probs[:, 1]
    probabilit_list.append(probs)
    calculate_f1_score(dtc, X, y)

    print("Predicting 4")
    dtc = pickle.load(open('../model/' + data_set + '/kNN_model_4', 'rb'))
    probs = dtc.predict_proba(X)
    probs = probs[:, 1]
    probabilit_list.append(probs)
    calculate_f1_score(dtc, X, y)

    color_list = ['r', 'b', 'm', 'y']
    label_list = ['k = 25', 'k = 150', 'k = 225', 'k = 300']

    plt = multiple_precision_recall_curves(y, probabilit_list, color_list, label_list)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.5, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('k Nearest Neighbors Precision-Recall Curve')
    plt.legend(loc="best")

    # plt.show()
    save_figure(plt, path + "plot/" + data_set, 'kNN_pr_curve.png')


def test_kNN_loan(X, y, path):
    data_set = 'loan'

    print("Predicting 1")
    probabilit_list = []
    dtc = pickle.load(open('../model/' + data_set + '/kNN_model_1', 'rb'))
    probs = dtc.predict_proba(X)
    probs = probs[:, 1]
    probabilit_list.append(probs)
    calculate_f1_score(dtc, X, y)

    print("Predicting 2")
    dtc = pickle.load(open('../model/' + data_set + '/kNN_model_2', 'rb'))
    probs = dtc.predict_proba(X)
    probs = probs[:, 1]
    probabilit_list.append(probs)
    calculate_f1_score(dtc, X, y)

    print("Predicting 3")
    dtc = pickle.load(open('../model/' + data_set + '/kNN_model_3', 'rb'))
    probs = dtc.predict_proba(X)
    probs = probs[:, 1]
    probabilit_list.append(probs)
    calculate_f1_score(dtc, X, y)

    print("Predicting 4")
    dtc = pickle.load(open('../model/' + data_set + '/kNN_model_4', 'rb'))
    probs = dtc.predict_proba(X)
    probs = probs[:, 1]
    probabilit_list.append(probs)
    calculate_f1_score(dtc, X, y)

    color_list = ['r', 'b', 'm', 'y']
    label_list = ['k = 5', 'k = 10', 'k = 25', 'k = 100']

    plt = multiple_precision_recall_curves(y, probabilit_list, color_list, label_list)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('k Nearest Neighbors Precision-Recall Curve')
    plt.legend(loc="best")

    # plt.show()
    save_figure(plt, path + "plot/" + data_set, 'kNN_pr_curve.png')
