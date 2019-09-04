import pickle
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score

x_test_file = open('../data/test/x', 'rb')
y_test_file = open('../data/test/y', 'rb')
x_test = pickle.load(x_test_file)
y_test = pickle.load(y_test_file)

boosted_model = pickle.load(open('../model/best_boosted_model', 'rb'))
boosted_model_results = boosted_model.predict(x_test)
boosted_model_average_precision = average_precision_score(y_test, boosted_model_results)
print("Boosted Decision Tree Results: ", boosted_model_average_precision)

dtc = pickle.load(open('../model/best_decision_tree_model', 'rb'))
dtc_results = dtc.predict(x_test)
dtc_average_precision = average_precision_score(y_test, dtc_results)
print("Decision Tree Results: ", dtc_average_precision)

kNN = pickle.load(open('../model/best_k_NN_model', 'rb'))
kNN_results = kNN.predict(x_test)
kNN_average_precision = average_precision_score(y_test, kNN_results)
print("k NN Results: ", kNN_average_precision)

neural_net = pickle.load(open('../model/best_Neural_Network_model', 'rb'))
neural_net_results = neural_net.predict(x_test)
neural_net_average_precision = average_precision_score(y_test, neural_net_results)
print("Neural Net Results: ", neural_net_average_precision)

svn = pickle.load(open('../model/best_SVN_model', 'rb'))
svn_results = svn.predict(x_test)
svn_average_precision = average_precision_score(y_test, svn_results)
print("SVN Results: ", svn_average_precision)
# neural_net_results
# cm = confusion_matrix(y_test, neural_net_results)
# cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
# print(cm)
x_test_file.close()
y_test_file.close()