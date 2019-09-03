from sklearn.neighbors import NearestNeighbors
import pickle

# different learning curves for different
def calculate_results(list, labels):
    results = []

    for i in range(len(list)):
        voting_result = vote(list[i], labels)
        results.append(voting_result)
    return results


def vote(x, labels):
    positive_count = 0
    negative_count = 0
    for i in range(len(x)):
        if labels[x[i]] == 0:
            negative_count += 1
        else:
            positive_count += 1
    if positive_count > negative_count:
        return 1
    elif negative_count > positive_count:
        return 0
    print("should not be here")


x_train_file = open('../data/train/x', 'rb')
y_train_file = open('../data/train/y', 'rb')
x_test_file = open('../data/test/x', 'rb')
y_test_file = open('../data/test/y', 'rb')
x_train = pickle.load(x_train_file)
y_train = pickle.load(y_train_file)
x_test = pickle.load(x_test_file)
y_test = pickle.load(y_test_file)

model = NearestNeighbors(n_neighbors=201, algorithm='ball_tree').fit(x_train)
distances, indices = model.kneighbors(x_test)

# print(indices)
results = calculate_results(indices, y_train)

correct = 0
incorrect = 0
for i in range(len(y_test)):
    if y_test[i] == results[i]:
        correct += 1
    else:
        incorrect += 1
print(correct / (correct + incorrect))

x_train_file.close()
y_train_file.close()
x_test_file.close()
y_test_file.close()
print("done")