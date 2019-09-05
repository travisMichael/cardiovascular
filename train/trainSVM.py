# https://scikit-learn.org/stable/modules/svm.html
from sklearn import svm
from utils import save_model
import pickle


def train_svm(dataset, path):
    print('Training Support Vector Machine...')
    model = svm.SVC(kernel='linear', cache_size=400)
    # model = svm.SVC(kernel='rbf', gamma=0.001, cache_size=400)

    x_train_file = open(path + 'data/train/x', 'rb')
    y_train_file = open(path + 'data/train/y', 'rb')
    x_train = pickle.load(x_train_file)
    y_train = pickle.load(y_train_file)

    model.fit(x_train, y_train)
    # result = model.predict(x_test)

    save_model(model, dataset, 'best_SVN_model')

    x_train_file.close()
    y_train_file.close()
    print("done")


if __name__ == "__main__":
    print('hello')
    train_svm('cardio', '../')
