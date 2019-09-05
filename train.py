import sys
from train.trainDecisionTree import train_dtc
from train.train_k_NearestNeighbors import train_k_NN
from train.trainBoostedDecisionTree import train_boosted_dtc
from train.trainSVM import train_svm
from train.trainNeuralNetwork import train_neural_net

if __name__ == "__main__":
    print(len(sys.argv))
    if len(sys.argv) < 3:
        print("Please specify filename, data set, and model")
    else:
        data_set = sys.argv[1]
        model = sys.argv[2]
        if model == 'kNN':
            train_k_NN(data_set, '')
        if model == 'Boosted':
            train_boosted_dtc(data_set, '')
        if model == 'DTC':
            train_dtc(data_set, '')
        if model == 'NeuralNet':
            train_neural_net(data_set, '')
        if model == 'svm':
            train_svm(data_set, '')
        if model == 'all':
            train_k_NN(data_set, '')
            train_boosted_dtc(data_set, '')
            train_dtc(data_set, '')
            train_neural_net(data_set, '')
            train_svm(data_set, '')

