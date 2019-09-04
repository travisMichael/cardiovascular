# from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from visualization_utils import multiple_learning_curves_plot
import pickle
import os


x_train_file = open('../data/train/x', 'rb')
y_train_file = open('../data/train/y', 'rb')
x_test_file = open('../data/test/x', 'rb')
y_test_file = open('../data/test/y', 'rb')
x_train = pickle.load(x_train_file)
y_train = pickle.load(y_train_file)
x_test = pickle.load(x_test_file)
y_test = pickle.load(y_test_file)

# model = NearestNeighbors(n_neighbors=201, algorithm='ball_tree').fit(x_train)
# distances, indices = model.kneighbors(x_test)
# model = KNeighborsClassifier(n_neighbors=200).fit(x_train, y_train)
# results = model.predict(x_test)


model_2 = KNeighborsClassifier(n_neighbors=75)
model_3 = KNeighborsClassifier(n_neighbors=150)
model_4 = KNeighborsClassifier(n_neighbors=225)
model_5 = KNeighborsClassifier(n_neighbors=300)
model_6 = KNeighborsClassifier(n_neighbors=50).fit(x_train, y_train)
# plt = multiple_learning_curves_plot(
#     [model_2],
#     x_train, y_train,
#     ["r", "y", "b", "g", "m"],
#     ["Max depth = 3", "Max depth = 4", "Max depth = 5", "Max depth = 6", "Max depth = 7"]
# )
#
# plt.title("Title")
# plt.xlabel("Training examples")
# plt.ylabel("Score")
# plt.grid()
#
# plt.legend(loc="best")
# plt.show()


# print(indices)
# results = calculate_results(indices, y_train)
#
# correct = 0
# # incorrect = 0
# # for i in range(len(y_test)):
# #     if y_test[i] == results[i]:
# #         correct += 1
# #     else:
# #         incorrect += 1
# # print(correct / (correct + incorrect))

if not os.path.exists('../model'):
    os.makedirs('../model')

pickle.dump(model_6, open("../model/best_k_NN_model", 'wb'))

x_train_file.close()
y_train_file.close()
x_test_file.close()
y_test_file.close()
print("done")