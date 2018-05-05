from MultiLayerNN import NeuralNetwork
# from MultiLayerNN_RELU import NeuralNetwork
# from MultiLayerNN_softmax import NeuralNetwork
from sklearn.externals import joblib
import h5py
import numpy as np
import h5py
import idx2numpy
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold



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


# print "!!!!!!!!!!y!!!!!!!!!!: ", y

def accuracy_score(y_set, predicted):
    n = len(y_set)
    matched = 0.0
    for indx in range(n):
        if y_set[indx] == predicted[indx]:
            matched += 1

    acc = (matched / n)
    return acc


X_test, y_test = loadAndProcesstXY(["mnist/t10k-images.idx3-ubyte", "mnist/t10k-labels.idx1-ubyte"])

# For sigmoid in output layer
# clf = NeuralNetwork(hidden_layer_sizes=[100, 50], max_iters=100, learning_rate_init=0.0000015)
# lr = 0.000025
# epochs = 3000 #0.0009 500, 0.00001 1000, 0.00001 2000, 0.00001 2500, 0.00005 2000, 0.00025 2000
# print "relu"
# print "learning rate: ", lr
# print "epochs: ", epochs
#
# # For softmax
# clf = NeuralNetwork(hidden_layer_sizes=[100, 50], max_iters=epochs, learning_rate_init=lr)
# clf.fit(X, y)
# predicted = clf.predict(X_test)
# print "predicted: ", predicted
# print "y_test: ", y_test
# print "relu"
# print "learning rate: ", lr
# print "epochs: ", epochs
# score = accuracy_score(y_test, predicted)
# print ""
# print "accuracy for mnist: ", score
#
# joblib.dump(clf, '/home/nehaj/Desktop/HW-3-NN/Weights/sigmoid_MNIST_complete_4')
# # joblib.dump(clf, '/home/nehaj/Desktop/HW-3-NN/Weights/relu_MNIST_complete')

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
    if K == 1:
        clf = NeuralNetwork(hidden_layer_sizes=[100, 50], max_iters=epochs, learning_rate_init=lr)
        model = clf.fit(X, y)
        joblib.dump(model, 'Weights/Model4_other')
        # joblib.dump(clf, '/home/nehaj/Desktop/HW-3-NN/Weights/relu_MNIST_complete')
        predicted = clf.predict(X_test)
        print "predicted: ", predicted
        print "y_test: ", y_test
        print "softmax on outer, sig on others"
        print "learning rate: ", lr
        print "epochs: ", epochs
        avg_score = accuracy_score(y_test, predicted)

    else:
        for k in xrange(0, K):
            x_train_set, x_test_set = partition_data(X, k, length / K)
            y_train_set, y_test_set = partition_data(y, k, length / K)
            nn = NeuralNetwork(hidden_layer_sizes=[100, 50], max_iters=epochs, learning_rate_init=lr)
            nn.fit(x_train_set, y_train_set)
            predicted = nn.predict(x_test_set)
            print "predicted: ", predicted
            print "y_test: ", y_test_set
            print "softmax with sigmoid"
            print "learning rate: ", lr
            print "epochs: ", epochs
            score = accuracy_score(y_test_set, predicted)
            avg_score += score
            # print(acc)
        avg_score /= K
    print("Mean accuracy is: ", avg_score)
    return avg_score



num_folds = 1
# Parameters for grid search: learning rate and max_iters
mnist_accuracy_list = list()
# epochs = [150, 200, 250]
# learn_rate = [0.001, 0.00085]
# epochs = [2000]
# learn_rate = [0.000003]
epochs = [200]
# learn_rate = [0.000008]
learn_rate = [0.0001]

print "relu big"
print "learning rate: ", learn_rate
print "epochs: ", epochs

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
# plt.ylabel('Accuracy score')
# plt.xlabel('Parameters: Epochs, Learning rate')
# plt.xticks(graph_x_dt, param_comb_list_dt, rotation=90)
# plt.plot(graph_x_dt, graph_y_dt)
# plt.tight_layout()
# # plt.show()
# plt.savefig("Plots/sigmoid-softmax_mnist_self.png")

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



