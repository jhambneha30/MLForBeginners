import numpy as np
import math
import operator
import h5py
from sklearn.externals import joblib

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

def predict_sigmoid(X_test, y_test):
    clf = joblib.load('Weights/Model1')
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
            z = sigmoid(np.dot(layer_input, wts[idx]) + bs[idx])
        a.append(z)
        layer_input = z

    last_output = a[count_iters - 1]
    y_computed = compute_result(last_output)
    score = accuracy_score(y_test, y_computed)
    print "accuracy on saved model: ", score


predict_sigmoid(X, y)


