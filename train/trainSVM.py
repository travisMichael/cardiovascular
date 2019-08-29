# https://scikit-learn.org/stable/modules/svm.html
from sklearn import svm
import pickle


model = svm.LinearSVC()

x_train_file = open('../data/train/x', 'rb')
y_train_file = open('../data/train/y', 'rb')
x_test_file = open('../data/test/x', 'rb')
y_test_file = open('../data/test/y', 'rb')
x_train = pickle.load(x_train_file)
y_train = pickle.load(y_train_file)
x_test = pickle.load(x_test_file)
y_test = pickle.load(y_test_file)

model.fit(x_train, y_train)
result = model.predict(x_test)

correct = 0
incorrect = 0
for i in range(len(y_test)):
    if y_test[i] == result[i]:
        correct += 1
    else:
        incorrect += 1
print(correct / (correct + incorrect))

x_train_file.close()
y_train_file.close()
x_test_file.close()
y_test_file.close()
print("done")