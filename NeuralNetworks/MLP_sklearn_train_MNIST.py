import numpy as np
from sklearn.externals import joblib
import h5py
import idx2numpy
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier


# Load the test data
def read_h5(filename):
	with h5py.File(filename, 'r') as hf:
		X = hf['X'][:]
		Y = hf['Y'][:]
	return X, Y

def load_data(filename):
	if ".h5" in filename[0]:
		X,Y = read_h5(filename[0])
	else:
		X = idx2numpy.convert_from_file(filename[0])
		Y = idx2numpy.convert_from_file(filename[1])
	return X,Y

def loadAndProcesstXY(filename):
	X,Y = load_data(filename)
	new_X,new_Y = [],[]
	for xx in range(len(X)):
		curr_x = np.ndarray.flatten(X[xx])
		curr_x_f = curr_x.astype(np.float)
		curr_x_n = np.divide(curr_x_f,255)
		curr_x_n_short = np.around(curr_x_n,decimals = 3)
		#print curr_x_n_short
		new_X.append(curr_x_n_short)
		new_Y.append([Y[xx]])
	X = np.array(new_X)#np.divide(np.array(new_X),255)
	# Y = np.array(new_Y)
	print "Shape of X:",X.shape
	print "Shape of Y:",Y.shape
	return X,Y


# Load test data
# X, y = load_data('dataset_partA.h5')
X, y = loadAndProcesstXY(["mnist/train-images.idx3-ubyte", "mnist/train-labels.idx1-ubyte"])
def accuracy_score(y_set, predicted):
    n = len(y_set)
    matched = 0.0
    for indx in range(n):
        if y_set[indx] == predicted[indx]:
            matched += 1

    acc = (matched / n)

    return acc

X_test, y_test = loadAndProcesstXY(["mnist/t10k-images.idx3-ubyte", "mnist/t10k-labels.idx1-ubyte"])

# clf = NeuralNetwork(hidden_layer_sizes=[100, 50], max_iters=30, learning_rate_init=0.0001)
# clf.fit(X, y)
# lr = 0.000005
# epochs = 2000
# k = 2
# kf = KFold(n_splits=k)
# avg_score = 0.0
# clf = NeuralNetwork(hidden_layer_sizes=[100, 50], random_state=1, max_iters=epochs, learning_rate_init=lr)
# for train_indices, test_indices in kf.split(X):
#     model = clf.fit(X[train_indices], y[train_indices])
#     predicted = clf.predict(X[test_indices])
#     score = accuracy_score(y[test_indices], predicted)
#     print "score for this fold: ", score
#     avg_score += score
# avg_score /= k
# print "normal"
# print "learning rate: ", lr
# print "epochs: ", epochs
# print "wts updated---average score: ", avg_score
#
# joblib.dump(model, '/home/nehaj/Desktop/HW-3-NN/Weights/sigmoid_subset')
# joblib.dump(clf, '/home/nehaj/Desktop/HW-3-NN/Weights/relu_subset')

# -----------------------GRID SEARCH WITH K FOLD--------------BEGIN-----------------------
# Pre process data and split it
def partition_data(x_data, k, fold_size):
    start = k * fold_size
    end = start + fold_size
    train = np.concatenate((x_data[:start], x_data[end:]))
    test = x_data[start:end]
    # print(test)
    # print(train)
    return train, test

def neural_nets(K, solver, lr):
    length = len(X)
    avg_score = 0.0
    for k in xrange(0, K):
        if K == 1:
            mlp = MLPClassifier(hidden_layer_sizes=[200, 40], verbose=True)
            mlp.fit(X, y)
            score = mlp.score(X_test, y_test)
            print(score)
            avg_score += score
            print "============================================="
            print "softmax on outer, sig on others"
            print "learning rate: ", lr
            print "solver: ", solver
            print "============================================="
        else:
            x_train_set, x_test_set = partition_data(X, k, length / K)
            y_train_set, y_test_set = partition_data(y, k, length / K)
            mlp = MLPClassifier(hidden_layer_sizes=[190, 10], learning_rate_init=lr, verbose=True)
            mlp.fit(x_train_set, y_train_set)
            score = mlp.score(x_test_set, y_test_set)
            print(score)
            avg_score += score
            # print "predicted: ", predicted
            print "============================================="
            print "relu"
            print "learning rate: ", lr
            print "solver: ", solver
        avg_score /= K
    print("Mean accuracy is: ", avg_score)
    print "============================================="
    return avg_score


num_folds = 1
# Parameters for grid search: learning rate and max_iters
mnist_accuracy_list = list()
solver = ['adam']
learn_rate = [0.001]

# define the grid here
parameters = [(s, lr) for s in solver for lr in learn_rate]

# do the grid search with k fold cross validation
# For making plots, we need 3 lists: graph_x, graph_y and param_comb_list
graph_x_dt, graph_y_dt, param_comb_list_dt = list(), list(), list()
index_dt = 1
for (s, lr) in parameters:
    param_comb = [s, lr]
    accuracy = neural_nets(num_folds, s, lr)
    mnist_accuracy_list.append(accuracy)
    graph_x_dt.append(index_dt)
    graph_y_dt.append(accuracy)  # this is same as dt_accuracy_list
    param_comb_list_dt.append(param_comb)
    index_dt += 1

# creating and saving the plot
# plt.ylabel('Accuracy score')
# plt.xlabel('Parameters: Solver, Learning rate')
# plt.xticks(graph_x_dt, param_comb_list_dt, rotation=90)
# plt.plot(graph_x_dt, graph_y_dt)
# plt.tight_layout()
# plt.show()
# plt.savefig("sigmoid_MNIST_MLP.png")
# print "acc list: ", mnist_accuracy_list

# save the best model and print the results
# max_accuracy = max(mnist_accuracy_list)
# max_accuracy_index = mnist_accuracy_list.index(max_accuracy)
# best_lr = parameters[max_accuracy_index][1]
# best_solver = parameters[max_accuracy_index][0]
# print("DT: max accuracy:", max_accuracy)
# print("DT: best lr", best_lr)
# print("DT: best solver", best_solver)

# -----------------------GRID SEARCH WITH K FOLD--------------ENDS-----------------------


# ======================================To use the saved models: TESTING==============================================
# Y_result = predict_sigmoid(X_test)
# print "predicted: ", Y_result
# print "y_test: ", y_test
# score = accuracy_score(y_test, Y_result)
# print "accuracy on saved model: ", score

