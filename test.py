import sys
from test.testModels import test_model
from test.plotRandomizedOptimization import plot_model


if __name__ == "__main__":
    print(len(sys.argv))
    if len(sys.argv) < 2:
        print("Please specify filename, data set, and model")
    elif len(sys.argv) == 2:
        plot_model('', 'all')
    else:
        data_set = sys.argv[1]
        model = sys.argv[2]
        test_model(model, '', data_set)

