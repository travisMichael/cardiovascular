from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import numpy as np
import matplotlib.pyplot as plt
import time

# The learning_curve_plot function is an adapted version from:
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html


def multiple_learning_curves_plot(model_list, x, y, colors, training_labels):
    number_of_models = len(model_list)
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    plt.figure()

    for i in range(number_of_models):
        print("Generating learning curve for model: " + str(i))
        start_time = time.time()
        model = model_list[i]
        curve_color = colors[i]
        label = training_labels[i]

        train_sizes, train_scores, test_scores = learning_curve(model, x, y, cv=cv, n_jobs=2, train_sizes=np.linspace(.1, 1.0, 5))

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color=curve_color)
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color=curve_color)
        plt.plot(train_sizes, train_scores_mean, 'o-', color=curve_color, label=label)
        plt.plot(train_sizes, test_scores_mean, 'o-', color=curve_color, linestyle='dashed')
        end_time = time.time() - start_time
        print("Learning curve finished for model: " + str(i) + " " + str(end_time))
    return plt
