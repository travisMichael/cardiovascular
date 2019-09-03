# https://statinfer.com/204-3-10-pruning-a-decision-tree-in-python/


from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
import pickle

N_CLASSES = np.unique([0 , 1])

# model = tree.DecisionTreeClassifier() max_depth=4
model = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=3), )

x_train_file = open('../data/train/x', 'rb')
y_train_file = open('../data/train/y', 'rb')
x_test_file = open('../data/test/x', 'rb')
y_test_file = open('../data/test/y', 'rb')
x_train = pickle.load(x_train_file)
y_train = pickle.load(y_train_file)
x_test = pickle.load(x_test_file)
y_test = pickle.load(y_test_file)

model.fit(x_train, y_train)
# model.partial_fit(x_train, y_train, classes=N_CLASSES)
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