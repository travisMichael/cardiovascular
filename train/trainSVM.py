# https://scikit-learn.org/stable/modules/svm.html
from sklearn import svm
from utils import save_model, load_data, train_and_time


def train_svm(path, with_plots):
    data_set = 'cardio'
    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')

    if not with_plots:
        print('Training SVM...')
        model_1 = train_and_time(svm.SVC(kernel='linear', cache_size=400), x_train, y_train)
        model_2 = train_and_time(svm.SVC(kernel='poly', cache_size=400), x_train, y_train)

        save_model(model_1, path + 'model/' + data_set, 'svm_model_1')
        save_model(model_2, path + 'model/' + data_set, 'svm_model_2')
    else:
        print('Training Support Vector Machine...')
        model = svm.SVC(kernel='linear', cache_size=400)
        # model = svm.SVC(kernel='rbf', gamma=0.001, cache_size=400)

        model.fit(x_train, y_train)
        # result = model.predict(x_test)

        save_model(model, path + 'model/' + data_set, 'best_SVN_model')
        print("done")


def train_svm_loan(path, with_plots):
    data_set = 'loan'
    x_train, y_train = load_data(path + 'data/' + data_set + '/train/')

    if not with_plots:
        print("Training SVM")
        model_1 = train_and_time(svm.SVC(kernel='linear', cache_size=400), x_train, y_train)
        # model_2 = train_and_time(svm.SVC(kernel='poly', cache_size=400), x_train, y_train)

        save_model(model_1, path + 'model/' + data_set, 'svm_model_1')
        # save_model(model_2, path + 'model/' + data_set, 'svm_model_2')
    else:
        print('Training Support Vector Machine...')
        model = svm.SVC(kernel='linear', cache_size=400)
        # model = svm.SVC(kernel='rbf', gamma=0.001, cache_size=400)

        model.fit(x_train, y_train)
        # result = model.predict(x_test)

        save_model(model, path + 'model/' + data_set, 'best_SVN_model')
        print("done")


if __name__ == "__main__":
    train_svm('../', False)
    # train_svm_loan('../', False)
