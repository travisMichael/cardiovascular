import numpy as np
import pickle
import os
from visualization_utils import multiple_learning_curves_plot
from sklearn import tree
# from joblib import dump

N_CLASSES = np.unique([0 , 1])

# model_1 = tree.DecisionTreeClassifier()

# max_leaf_nodes=8

x_train_file = open('../data/train/x', 'rb')
y_train_file = open('../data/train/y', 'rb')
x_test_file = open('../data/test/x', 'rb')
y_test_file = open('../data/test/y', 'rb')
x_train = pickle.load(x_train_file)
y_train = pickle.load(y_train_file)
x_test = pickle.load(x_test_file)
y_test = pickle.load(y_test_file)

# model_2 = tree.DecisionTreeClassifier(max_depth=3)
# model_3 = tree.DecisionTreeClassifier(max_depth=4)
# model_4 = tree.DecisionTreeClassifier(max_depth=5)
# model_5 = tree.DecisionTreeClassifier(max_depth=6)
# model_6 = tree.DecisionTreeClassifier(max_depth=7)
# plt = multiple_learning_curves_plot(
#     [model_2, model_3, model_4, model_5, model_6],
#     x_train, y_train,
#     ["r", "y", "b", "g", "m"],
#     ["Max depth = 3", "Max depth = 4", "Max depth = 5", "Max depth = 6", "Max depth = 7"]
# )

model_2 = tree.DecisionTreeClassifier(max_depth=5, max_leaf_nodes=60)
model_3 = tree.DecisionTreeClassifier(max_depth=5, max_leaf_nodes=50)
model_4 = tree.DecisionTreeClassifier(max_depth=5, max_leaf_nodes=40)
model_5 = tree.DecisionTreeClassifier(max_depth=5, max_leaf_nodes=30)
model_6 = tree.DecisionTreeClassifier(max_depth=5, max_leaf_nodes=20)
model_5.fit(x_train, y_train)
# plt = multiple_learning_curves_plot(
#     [model_2, model_3, model_4, model_5, model_6],
#     x_train, y_train,
#     ["r", "y", "b", "g", "m"],
#     ["Max depth = 3", "Max depth = 4", "Max depth = 5", "Max depth = 6", "Max depth = 7"]
# )



# plt.title("Title")
# plt.xlabel("Training examples")
# plt.ylabel("Score")
# plt.grid()
#
# plt.legend(loc="best")
# plt.show()
if not os.path.exists('../model'):
    os.makedirs('../model')

pickle.dump(model_5, open("../model/best_decision_tree_model", 'wb'))

x_train_file.close()
y_train_file.close()
x_test_file.close()
y_test_file.close()
print("done")