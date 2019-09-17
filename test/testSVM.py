import pickle
from sklearn.metrics import average_precision_score
from visualization_utils import multiple_precision_recall_curves
from utils import save_figure


def test_svm(X, y, path):
    data_set = 'cardio'

    print("Predicting 1")
    probabilit_list = []
    dtc = pickle.load(open('../model/' + data_set + '/svm_model_1', 'rb'))
    probs = dtc.predict_proba(X)
    probs = probs[:, 1]
    probabilit_list.append(probs)

    print("Predicting 2")
    dtc = pickle.load(open('../model/' + data_set + '/svm_model_2', 'rb'))
    probs = dtc.predict_proba(X)
    probs = probs[:, 1]
    probabilit_list.append(probs)

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
    dtc_average_precision = average_precision_score(y, dtc.predict(X))
    print("Neural Ne Results: ", dtc_average_precision)
