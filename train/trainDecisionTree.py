import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

N_CLASSES = np.unique([0 , 1])

model = tree.DecisionTreeClassifier()
# model = tree.DecisionTreeClassifier(max_depth=3)
# model = tree.DecisionTreeClassifier() max_leaf_nodes=8

x_train_file = open('../data/train/x', 'rb')
y_train_file = open('../data/train/y', 'rb')
x_test_file = open('../data/test/x', 'rb')
y_test_file = open('../data/test/y', 'rb')
x_train = pickle.load(x_train_file)
y_train = pickle.load(y_train_file)
x_test = pickle.load(x_test_file)
y_test = pickle.load(y_test_file)

# model.fit(x_train, y_train)
# model.partial_fit(x_train, y_train, classes=N_CLASSES)
# result = model.predict(x_test)
#
# correct = 0
# incorrect = 0
# for i in range(len(y_test)):
#     if y_test[i] == result[i]:
#         correct += 1
#     else:
#         incorrect += 1
# print(correct / (correct + incorrect))

cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

train_sizes, train_scores, test_scores = learning_curve(
    model, x_train, y_train, cv=cv, n_jobs=2, train_sizes=np.linspace(.1, 1.0, 5))

plt.figure()
plt.title("Title")
# if ylim is not None:
#     plt.ylim(*ylim)
plt.xlabel("Training examples")
plt.ylabel("Score")
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.grid()

plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1,
                 color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
         label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
         label="Cross-validation score")

plt.legend(loc="best")
plt.show()
x_train_file.close()
y_train_file.close()
x_test_file.close()
y_test_file.close()
print("done")