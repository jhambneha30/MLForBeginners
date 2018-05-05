import os
import os.path
import argparse
import h5py
import numpy as np
from sklearn.utils import shuffle
    from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.naive_bayes import GaussianNB
# from Models.GaussianNB import GaussianNB
from sklearn.linear_model import LogisticRegression
# from Models.LogisticRegression import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
# from Models.DecisionTreeClassifier import DecisionTreeClassifier

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type = str)
parser.add_argument("--weights_path", type = str)
parser.add_argument("--train_data", type = str)
parser.add_argument("--plots_save_dir", type = str)

args = parser.parse_args()

MAXIMUM_ACCURACY = -10
MAX_ACCURACY_MODEL = None
# Load the test data
def load_h5py(filename):
    with h5py.File(filename, 'r') as hf:
        X = hf['X'][:]
        Y = hf['Y'][:]
    return X, Y

# Pre process data and split it
def partition_data(x_data, k, fold_size):
    start = k * fold_size
    end = start + fold_size
    train = np.concatenate((x_data[:start], x_data[end:]))
    test = x_data[start:end]
    # print(test)
    # print(train)
    return train, test

X, Y = load_h5py(args.train_data)
# X, Y = load_h5py('/home/nehaj/Desktop/MLhomework1_MT16037/Template/Data/part_A_train.h5')
# print(X.shape)
# print(Y.shape)
# X = list(X)
# Y = list(Y)
# create training and testing vars
labels = list()
for l in Y:
    i = np.where(l == 1)[0][0]
    labels.append(i)

# print labels
data_x, data_y = shuffle(X, labels, random_state=0)
# print(datage_x)
length = len(data_x)

def gaussian(K):
    mean_accuracy = 0
    for k in xrange(0, K):
        x_train_set, x_test_set = partition_data(data_x, k, length / K)
        y_train_set, y_test_set = partition_data(data_y, k, length / K)
        gb = GaussianNB()
        gb.fit(x_train_set, y_train_set)
        predicted = gb.predict(x_test_set)
        acc = accuracy_score(y_test_set, predicted)
        mean_accuracy += acc
        # print(acc)
    mean_accuracy /= K
    print("Gaussian: Mean accuracy is: ", mean_accuracy)
    return mean_accuracy


def logistic_regression(K, param_it, param_c):
    mean_accuracy = 0
    for k in xrange(0, K):
        x_train_set, x_test_set = partition_data(data_x, k, length / K)
        y_train_set, y_test_set = partition_data(data_y, k, length / K)
        lr = LogisticRegression(C=param_c, max_iter=param_it)
        lr.fit(x_train_set, y_train_set)
        predicted = lr.predict(x_test_set)
        acc = accuracy_score(y_test_set, predicted)
        mean_accuracy += acc
        # print(acc)
    mean_accuracy /= K
    # print("LR: Mean accuracy is: ", mean_accuracy)
    return mean_accuracy


def decision_trees(K, param_md, param_mss):
    mean_accuracy = 0
    for k in xrange(0, K):
        x_train_set, x_test_set = partition_data(data_x, k, length / K)
        y_train_set, y_test_set = partition_data(data_y, k, length / K)
        # if args.model_name == 'GaussianNB':
        dt = DecisionTreeClassifier(max_depth=param_md, min_samples_split=param_mss)
        dt.fit(x_train_set, y_train_set)
        predicted = dt.predict(x_test_set)
        acc = accuracy_score(y_test_set, predicted)
        mean_accuracy += acc
        # print(acc)
    mean_accuracy /= K
    # print("DT: Mean accuracy is: ", mean_accuracy)
    return mean_accuracy
global best_max_depth, best_mss, best_iter, best_c
# Train the models
num_folds = 4

# if True:
if args.model_name == 'GaussianNB':
    accuracy_gb = gaussian(num_folds)
    if accuracy_gb > MAXIMUM_ACCURACY:
        MAXIMUM_ACCURACY = accuracy_gb
        MAX_ACCURACY_MODEL = 1

# if True:
elif args.model_name == 'LogisticRegression':
    # using c (regularization) and max_iters
    lr_accuracy_list = list()
    max_iters_list = [50, 80, 100]
    C_list = [0.01, 0.3, 0.5]
    # penalty_list = ['l1']

    # define the grid here
    parameters_lr = [(it, c) for it in max_iters_list
                  for c in C_list]

    # do the grid search with k fold cross validation
    # For making plots, we need 3 lists: graph_x, graph_y and param_comb_list
    graph_x_lr, graph_y_lr, param_comb_list_lr = list(), list(), list()
    index_lr = 1
    for (it, c) in parameters_lr:
        param_comb_lr = [it, c]
        accuracy_lr = logistic_regression(num_folds, it, c)
        lr_accuracy_list.append(accuracy_lr)
        graph_x_lr.append(index_lr)
        graph_y_lr.append(accuracy_lr)  # this is same as lr_accuracy_list
        param_comb_list_lr.append(param_comb_lr)
        index_lr += 1

    # creating and saving the plot
    plt.xticks(graph_x_lr, param_comb_list_lr, rotation=90)
    plt.plot(graph_x_lr, graph_y_lr)
    plt.tight_layout()
    # plt.show()
    plt.savefig(
        "" + args.plots_save_dir + "/" + args.train_data.replace("Data/", "") + "_" + args.model_name + ".png")

    # save the best model and print the results
    max_accuracy_lr = max(lr_accuracy_list)
    if max_accuracy_lr > MAXIMUM_ACCURACY:
        MAXIMUM_ACCURACY = max_accuracy_lr
        MAX_ACCURACY_MODEL = 2
    max_accuracy_index = lr_accuracy_list.index(max_accuracy_lr)
    best_iter = parameters_lr[max_accuracy_index][0]
    best_c = parameters_lr[max_accuracy_index][1]
    print("LR: max accuracy:", max_accuracy_lr)
    print("LR: best number of iterations", best_iter)
    print("LR: best C", best_c)

# if True:
elif args.model_name == 'DecisionTreeClassifier':
    # Taking max_depth and min_samples_split as the parameters on which grid search is to be applied
    dt_accuracy_list = list()
    min_samples_split = [20, 60, 80]
    max_depth = [1, 2, 3]

    # define the grid here
    parameters = [(mss, md) for mss in min_samples_split
                     for md in max_depth]

    # do the grid search with k fold cross validation
    # For making plots, we need 3 lists: graph_x, graph_y and param_comb_list
    graph_x_dt, graph_y_dt, param_comb_list_dt = list(), list(), list()
    index_dt = 1
    for (mss, md) in parameters:
        param_comb = [mss, md]
        accuracy = decision_trees(num_folds, md, mss)
        dt_accuracy_list.append(accuracy)
        graph_x_dt.append(index_dt)
        graph_y_dt.append(accuracy)  # this is same as dt_accuracy_list
        param_comb_list_dt.append(param_comb)
        index_dt += 1

    # creating and saving the plot
    plt.xticks(graph_x_dt, param_comb_list_dt, rotation=90)
    plt.plot(graph_x_dt, graph_y_dt)
    plt.tight_layout()
    # plt.show()
    plt.savefig(
        "" + args.plots_save_dir + "/" + args.train_data.replace("Data/", "") + "_" + args.model_name + ".png")

    # save the best model and print the results
    max_accuracy_dt = max(dt_accuracy_list)
    max_accuracy_index = dt_accuracy_list.index(max_accuracy_dt)
    best_max_depth = parameters[max_accuracy_index][1]
    best_mss = parameters[max_accuracy_index][0]
    print("DT: max accuracy:", max_accuracy_dt)
    print("DT: best_max_depth", best_max_depth)
    print("DT: best_min_sample_split", best_mss)
    if max_accuracy_dt > MAXIMUM_ACCURACY:
        MAXIMUM_ACCURACY = max_accuracy_dt
        MAX_ACCURACY_MODEL = 3
else:
    raise Exception("Invalid Model name")


# model = LogisticRegression(C=0.01, max_iter=50)
# model.fit(data_x, data_y)
# joblib.dump(model, '/home/nehaj/Desktop/MLhomework1_MT16037/Template/Weights/LR_partA')

# if MAX_ACCURACY_MODEL == 1:
#     print("Training the Gaussian model and saving it...")
#     model = GaussianNB()
#     model.fit(data_x, data_y)
#     joblib.dump(model, '/home/nehaj/Desktop/MLhomework1_12909/Template/Weights/GB_partB')
#
# elif MAX_ACCURACY_MODEL == 2:
#     print("Training the Logistic Regression model and saving it...")
#     model = LogisticRegression(C=best_c, max_iter=best_iter)
#     model.fit(data_x, data_y)
#     joblib.dump(model, '/home/nehaj/Desktop/MLhomework1_12909/Template/Weights/LR_partB')
#
# elif MAX_ACCURACY_MODEL == 3:
#     print("Training the Decision Tree model and saving it...")
#     model = DecisionTreeClassifier(max_depth=best_max_depth, min_samples_split=best_mss)
#     model.fit(data_x, data_y)
#     joblib.dump(model, '/home/nehaj/Desktop/MLhomework1_12909/Template/Weights/DT_partB')


