import sys
from train.trainDecisionTree import train_dtc
from train.train_k_NearestNeighbors import train_k_NN
from train.trainBoostedDecisionTree import train_boosted_dtc
from train.trainSVM import train_svm
from train.trainNeuralNetwork import train_neural_net

if __name__ == "__main__":
    print(len(sys.argv))
    if len(sys.argv) < 3:
        print("Please specify filename, dataset, and model")
    else:
        dataset = sys.argv[1]
        model = sys.argv[2]
        if model == 'kNN':
            train_k_NN(dataset, '')
        if model == 'Boosted':
            train_boosted_dtc(dataset, '')
        if model == 'DTC':
            train_dtc(dataset, '')
        if model == 'NeuralNet':
            train_neural_net(dataset, '')
        if model == 'svm':
            train_svm(dataset, '')
        if model == 'all':
            train_k_NN(dataset, '')
            train_boosted_dtc(dataset, '')
            train_dtc(dataset, '')
            train_neural_net(dataset, '')
            train_svm(dataset, '')

        print("done")
