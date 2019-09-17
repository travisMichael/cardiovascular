import pickle
from sklearn.metrics import average_precision_score
from utils import load_data
from visualization_utils import multiple_precision_recall_curves
from utils import save_figure


def test_neural_net_cardio(X, y, path):
    data_set = 'cardio'
    probabilit_list = []
    dtc = pickle.load(open('../model/' + data_set + '/neural_net_model_1', 'rb'))
    probs = dtc.predict_proba(X)
    probs = probs[:, 1]
    probabilit_list.append(probs)

    dtc = pickle.load(open('../model/' + data_set + '/neural_net_model_2', 'rb'))
    probs = dtc.predict_proba(X)
    probs = probs[:, 1]
    probabilit_list.append(probs)

    dtc = pickle.load(open('../model/' + data_set + '/neural_net_model_3', 'rb'))
    probs = dtc.predict_proba(X)
    probs = probs[:, 1]
    probabilit_list.append(probs)

    dtc = pickle.load(open('../model/' + data_set + '/neural_net_model_4', 'rb'))
    probs = dtc.predict_proba(X)
    probs = probs[:, 1]
    probabilit_list.append(probs)

    color_list = ['r', 'b', 'm', 'y']
    label_list = ['HLS = 5', 'HLS = 40', 'HLS = 5 x 5', 'HLS = 40 x 40']

    plt = multiple_precision_recall_curves(y, probabilit_list, color_list, label_list)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.5, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Neural Network with Varying Hidden Layer Size (HLS) \n Precision-Recall Curve')
    plt.legend(loc="best")

    # plt.show()
    save_figure(plt, path + "plot/" + data_set, 'neural_net_pr_curve.png')
    dtc_average_precision = average_precision_score(y, dtc.predict(X))
    print("Neural Ne Results: ", dtc_average_precision)


def test_neural_net_loan(X, y, path):
    data_set = 'loan'
    probabilit_list = []
    dtc = pickle.load(open('../model/' + data_set + '/neural_net_model_1', 'rb'))
    probs = dtc.predict_proba(X)
    probs = probs[:, 1]
    probabilit_list.append(probs)

    dtc = pickle.load(open('../model/' + data_set + '/neural_net_model_2', 'rb'))
    probs = dtc.predict_proba(X)
    probs = probs[:, 1]
    probabilit_list.append(probs)

    dtc = pickle.load(open('../model/' + data_set + '/neural_net_model_3', 'rb'))
    probs = dtc.predict_proba(X)
    probs = probs[:, 1]
    probabilit_list.append(probs)

    dtc = pickle.load(open('../model/' + data_set + '/neural_net_model_4', 'rb'))
    probs = dtc.predict_proba(X)
    probs = probs[:, 1]
    probabilit_list.append(probs)

    color_list = ['r', 'b', 'm', 'y']
    label_list = ['HLS = 20 x 5', 'HLS = 50 x 5', 'HLS = 100 x 5', 'HLS = 500 x 5']

    plt = multiple_precision_recall_curves(y, probabilit_list, color_list, label_list)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Neural Network with Varying Hidden Layer Size (HLS) \n Precision-Recall Curve')
    plt.legend(loc="best")

    # plt.show()
    save_figure(plt, path + "plot/" + data_set, 'neural_net_pr_curve.png')
    dtc_average_precision = average_precision_score(y, dtc.predict(X))
    print("Neural Ne Results: ", dtc_average_precision)
