import sys
from train.trainDecisionTree import train_dtc, train_dtc_loan
from train.train_k_NearestNeighbors import train_k_NN, train_k_NN_loan
from train.trainBoostedDecisionTree import train_boosted_dtc, train_boosted_dtc_loan
from train.trainSVM import train_svm, train_svm_loan
from train.trainNeuralNetwork import train_neural_net, train_neural_net_with_loan_data

if __name__ == "__main__":
    print(len(sys.argv))
    if len(sys.argv) < 3:
        print("Please specify filename, data set, and model")
    else:
        data_set = sys.argv[1]
        model = sys.argv[2]
        with_plots = sys.argv[3]
        if data_set == 'cardio':
            if model == 'kNN':
                train_k_NN('', with_plots)
            if model == 'Boosted':
                train_boosted_dtc('', with_plots)
            if model == 'DTC':
                train_dtc('', with_plots)
            if model == 'NeuralNet':
                train_neural_net('', with_plots)
            if model == 'svm':
                train_svm('', with_plots)
            if model == 'all':
                train_k_NN('', with_plots)
                train_boosted_dtc('', with_plots)
                train_dtc('', with_plots)
                train_neural_net('', with_plots)
                # train_svm('', with_plots)
        elif data_set == 'loan':
            if model == 'kNN':
                train_k_NN_loan('', with_plots)
            if model == 'Boosted':
                train_boosted_dtc_loan('', with_plots)
            if model == 'DTC':
                train_dtc_loan('', with_plots)
            if model == 'NeuralNet':
                train_neural_net_with_loan_data('', with_plots)
            if model == 'svm':
                train_svm_loan('', with_plots)
            if model == 'all':
                train_k_NN_loan('', with_plots)
                train_boosted_dtc_loan('', with_plots)
                train_dtc_loan('', with_plots)
                train_neural_net_with_loan_data('', with_plots)
                # train_svm_loan('', with_plots)
