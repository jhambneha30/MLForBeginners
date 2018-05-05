# from MultiLayerNN import NeuralNetwork
# from MultiLayerNN_softmax import NeuralNetwork
from MultiLayerNN_RELU import NeuralNetwork
import h5py
import numpy as np
from sklearn.externals import joblib
import h5py
import idx2numpy
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt




# Load the test data
def read_h5(filename):
	with h5py.File(filename, 'r') as hf:
		X = hf['X'][:]
		Y = hf['Y'][:]
	return X, Y


# Load test data
X, y = read_h5('dataset_partA.h5')

print "----y: ", y

def reshapeX(X):
	nsamples, nx, ny = X.shape
	X = X.reshape((nsamples, nx * ny))
	return X

def accuracy_score(y_set, predicted):
    n = len(y_set)
    matched = 0.0
    for indx in range(n):
        if y_set[indx] == predicted[indx]:
            matched += 1

    acc = (matched / n)

    return acc

# Converting 3D array X to 2D array X
X = reshapeX(X)
print np.shape(X)

for n, i in enumerate(y):
    if i == 7:
        y[n] = 0
    if i == 9:
        y[n] = 1


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

def neural_nets(K, epochs, lr):
    length = len(X)
    avg_score = 0.0
    for k in xrange(0, K):
        x_train_set, x_test_set = partition_data(X, k, length / K)
        y_train_set, y_test_set = partition_data(y, k, length / K)
        nn = NeuralNetwork(hidden_layer_sizes=[100, 50], max_iters=epochs, learning_rate_init=lr)
        model = nn.fit(x_train_set, y_train_set)
        joblib.dump(model, '/home/nehaj/Desktop/HW-3-NN/Weights/Model3')
        # joblib.dump(clf, '/home/nehaj/Desktop/HW-3-NN/Weights/relu_subset')
        predicted = nn.predict(x_test_set)
        print "predicted: ", predicted
        print "y_test: ", y_test_set
        print "relu"
        print "learning rate: ", lr
        print "epochs: ", epochs
        score = accuracy_score(y_test_set, predicted)
        avg_score += score
        print("score for this fold: ", score)
    avg_score /= K
    print("Mean accuracy is: ", avg_score)
    return avg_score



num_folds = 2
# Parameters for grid search: learning rate and max_iters
mnist_accuracy_list = list()
epochs = [2200]
learn_rate = [0.000008]

# define the grid here
parameters = [(e, lr) for e in epochs
                 for lr in learn_rate]

# do the grid search with k fold cross validation
# For making plots, we need 3 lists: graph_x, graph_y and param_comb_list
graph_x_dt, graph_y_dt, param_comb_list_dt = list(), list(), list()
index_dt = 1
for (e, lr) in parameters:
    param_comb = [e, lr]
    accuracy = neural_nets(num_folds, e, lr)
    mnist_accuracy_list.append(accuracy)
    graph_x_dt.append(index_dt)
    graph_y_dt.append(accuracy)  # this is same as dt_accuracy_list
    param_comb_list_dt.append(param_comb)
    index_dt += 1

# creating and saving the plot
plt.ylabel('Accuracy score')
plt.xlabel('Parameters: Epochs, Learning rate')
plt.xticks(graph_x_dt, param_comb_list_dt, rotation=90)
plt.plot(graph_x_dt, graph_y_dt)
plt.tight_layout()
plt.show()
plt.savefig("relu_subsetA_self.png")
print "acc list: ", mnist_accuracy_list

# save the best model and print the results
max_accuracy = max(mnist_accuracy_list)
max_accuracy_index = mnist_accuracy_list.index(max_accuracy)
best_lr = parameters[max_accuracy_index][1]
best_epochs = parameters[max_accuracy_index][0]
print("max accuracy:", max_accuracy)
print("best lr", best_lr)
print("best epochs", best_epochs)

# -----------------------GRID SEARCH WITH K FOLD--------------ENDS-----------------------


# ======================================To use the saved models: TESTING==============================================
# Y_result = predict_sigmoid(X_test)
# print "predicted: ", Y_result
# print "y_test: ", y_test
# score = accuracy_score(y_test, Y_result)
# print "accuracy on saved model: ", score

