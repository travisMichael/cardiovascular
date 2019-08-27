# https://scikit-learn.org/stable/modules/neural_networks_supervised.html
from sklearn.neural_network import MLPClassifier

X = [[0., 0.], [1., 1.], [-1., 1.], [0., 0.]]
y = [0, 1, 1, 1]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(120, 2), random_state=1, max_iter=1)


clf.fit(X, y)

prob_result = clf.predict_proba([[2., 2.], [1., 2.]])
print(prob_result)

clf.fit(X, y)
clf.fit(X, y)
clf.fit(X, y)

# result = clf.predict([[2., 2.], [-1., -2.]])
# print(result)

prob_result = clf.predict_proba([[2., 2.], [1., 2.]])
print(prob_result)