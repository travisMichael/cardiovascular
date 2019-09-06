import sys
from test.testModel import test_model


if __name__ == "__main__":
    print(len(sys.argv))
    if len(sys.argv) < 3:
        print("Please specify filename, data set, and model")
    else:
        data_set = sys.argv[1]
        model = sys.argv[2]
        test_model(model, '', data_set)

