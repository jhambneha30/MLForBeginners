import numpy as np
import math
import operator
from sklearn.externals import joblib
import h5py
import idx2numpy

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
# X, y = loadAndProcesstXY(["mnist/train-images.idx3-ubyte", "mnist/train-labels.idx1-ubyte"])
X_test, y_test = loadAndProcesstXY(["mnist/t10k-images.idx3-ubyte", "mnist/t10k-labels.idx1-ubyte"])

def accuracy_score(y_set, predicted):
    n = len(y_set)
    matched = 0.0
    for indx in range(n):
        if y_set[indx] == predicted[indx]:
            matched += 1

    acc = (matched / n)

    return acc

def compute_result(y_vector):
    y_op = list()
    for idx in range(len(y_vector)):
        max_index, max_value = max(enumerate(y_vector[idx]), key=operator.itemgetter(1))
        y_op.append(max_index)
    return y_op


def sigmoid(matrix):
    return 1.0 / (1.0 + np.exp(-matrix))


# Derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return x * (1 - x)


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1)[:, None]


def derivatives_softmax(x):
    return softmax(x) * (1 - softmax(x))


def relu(x):
    return np.maximum(x, 0)


def derivatives_relu(x):
    return 1. * (x > 0)

def predict_relu(X_test, y_test):
    clf = joblib.load('/home/nehaj/Desktop/HW-3-NN/Weights/Model4_other')
    # Helper function to predict an output (0 or 1)
    wts, bs = clf['weights'], clf['bias']
    count_iters = clf['ci']
    # Forward propagation
    a = []
    layer_input = X_test
    for idx in range(count_iters):
        if idx == count_iters - 1:
            z = softmax(np.dot(layer_input, wts[idx]) + bs[idx])
        else:
            z = relu(np.dot(layer_input, wts[idx]) + bs[idx])
            # a[idx] = logistic.cdf(z[idx])
        a.append(z)
        layer_input = z

    last_output = a[count_iters - 1]
    y_computed = compute_result(last_output)

    score = accuracy_score(y_test, y_computed)
    print "accuracy on saved model: ", score

predict_relu(X_test, y_test)


