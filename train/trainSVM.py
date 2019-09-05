# https://scikit-learn.org/stable/modules/svm.html
from sklearn import svm
from utils import save_model, load_data


def train_svm(data_set, path):
    print('Training Support Vector Machine...')
    model = svm.SVC(kernel='linear', cache_size=400)
    # model = svm.SVC(kernel='rbf', gamma=0.001, cache_size=400)

    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')

    model.fit(x_train, y_train)
    # result = model.predict(x_test)

    save_model(model, data_set, 'best_SVN_model')
    print("done")


if __name__ == "__main__":
    train_svm('cardio', '../')
