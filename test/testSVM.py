import pickle
from visualization_utils import multiple_precision_recall_curves
from utils import save_figure, calculate_f1_score


def test_svm(X, y, path):
    data_set = 'cardio'

    print("Predicting 1")
    probabilit_list = []
    dtc = pickle.load(open(path + 'model/' + data_set + '/svm_model_1', 'rb'))
    probs = dtc.predict_proba(X)
    probs = probs[:, 1]
    probabilit_list.append(probs)
    calculate_f1_score(dtc, X, y)

    print("Predicting 2")
    dtc = pickle.load(open(path + 'model/' + data_set + '/svm_model_2', 'rb'))
    probs = dtc.predict_proba(X)
    probs = probs[:, 1]
    probabilit_list.append(probs)
    calculate_f1_score(dtc, X, y)

    color_list = ['r', 'b']
    label_list = ['kernel = linear', 'kernel = polynomial']

    plt = multiple_precision_recall_curves(y, probabilit_list, color_list, label_list)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.5, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Support Vector Machine Precision-Recall Curve')
    plt.legend(loc="best")

    # plt.show()
    save_figure(plt, path + "plot/" + data_set, 'svm_pr_curve.png')


def test_svm_loan(X, y, path):
    data_set = 'loan'

    print("Predicting 1")
    probabilit_list = []
    dtc = pickle.load(open(path + 'model/' + data_set + '/f_svm_model_1', 'rb'))
    probs = dtc.predict_proba(X)
    probs = probs[:, 1]
    probabilit_list.append(probs)
    calculate_f1_score(dtc, X, y)

    print("Predicting 2")
    dtc = pickle.load(open(path + 'model/' + data_set + '/f_svm_model_2', 'rb'))
    probs = dtc.predict_proba(X)
    probs = probs[:, 1]
    probabilit_list.append(probs)
    calculate_f1_score(dtc, X, y)

    color_list = ['r', 'b']
    label_list = ['kernel = linear', 'kernel = polynomial']

    plt = multiple_precision_recall_curves(y, probabilit_list, color_list, label_list)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Support Vector Machine Precision-Recall Curve')
    plt.legend(loc="best")

    # plt.show()
    save_figure(plt, path + "plot/" + data_set, 'svm_pr_curve.png')
