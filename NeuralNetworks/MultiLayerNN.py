import numpy as np
import math
import operator
from random import seed
from scipy.stats import logistic
from random import random
from math import exp

class NeuralNetwork(object):

    def __init__(self, hidden_layer_sizes=(100, 50), random_state=None,  learning_rate_init=0.001, max_iters=100):
        # define all the model weights and state here
        self.hidden_layer_sizes = hidden_layer_sizes
        self.random_state = random_state
        self.learning_rate_init = learning_rate_init
        self.max_iters = max_iters
        self.ci = None
        self.nn_model = None

    def sigmoid(self, matrix):
        return 1.0 / (1.0 + np.exp(-matrix))

    # Derivative of Sigmoid Function
    def derivatives_sigmoid(self, x):
        return x * (1 - x)

    def build_model(self, X, y, hidden_layers, n_inputs, n_outputs, print_loss=True):
        # Initialize the parameters to random values. We need to learn these.
        # print y
        np.random.seed(self.random_state)
        num_hls = len(hidden_layers)
        final = None
        #Initializing weights for n hidden layers (n+1 weights required for n hidden layers)
        weights = {}
        bias = {}
        print "num inputs:", n_inputs
        W1 = np.random.randn(n_inputs, hidden_layers[0]) / np.sqrt(n_inputs)
        b1 = np.zeros((1, hidden_layers[0]))
        # print W1
        weights[0] = W1
        bias[0] = b1
        for h in range(num_hls-1):
            w = np.random.randn(hidden_layers[h], hidden_layers[h+1]) / np.sqrt(hidden_layers[h])
            b = np.zeros((1, hidden_layers[h+1]))
            weights[h+1] = w
            bias[h+1] = b
        W_last = np.random.randn(hidden_layers[num_hls - 1], n_outputs) / np.sqrt(hidden_layers[num_hls - 1])
        b_last = np.zeros((1, n_outputs))
        weights[num_hls] = W_last
        bias[num_hls] = b_last

        # print "weights: ", weights
        # print "biases: ", bias

        count_iters = num_hls + 1
        self.ci = count_iters
        # model is a dictionary that is returned
        model = {}
        # Gradient descent. For each batch...
        for e in xrange(0, self.max_iters):
            print "epoch: ", e
            # Forward propagation

            a = []
            layer_input = X
            for idx in range(count_iters):
                z = self.sigmoid(np.dot(layer_input, weights[idx]) + bias[idx])
                # a[idx] = logistic.cdf(z[idx])
                a.append(z)
                layer_input = z

            last_output = layer_input
            # print("wts length: ", len(weights))
            # print("activations length: ", len(a))

            # print "shape of last output: ", np.shape(last_output)
            # print "output: ", last_output

            # Back propagation

            E = y-last_output
            # print "iters:", count_iters
            right_lyr = last_output
            error_at_hl = E
            delta_op = []
            for lyr in range(count_iters-1, 0, -1):
                # print "lyr: ", lyr
                layer_slope = self.derivatives_sigmoid(right_lyr)
                d_output = np.multiply(error_at_hl, layer_slope)
                delta_op.append(d_output)
                error_at_hl = d_output.dot(weights[lyr].T)
                right_lyr = a[lyr-1]


            layer_slope = self.derivatives_sigmoid(a[0])
            d_output = np.multiply(error_at_hl, layer_slope)
            delta_op.append(d_output)
            for l in range(count_iters-1, 0, -1):
                # print "l: ", l
                del_ind = num_hls-l
                weights[l] += np.transpose(a[l-1]).dot(delta_op[del_ind]) * self.learning_rate_init
                bias[l] += np.sum(delta_op[del_ind], axis=0, keepdims=True) * self.learning_rate_init
            weights[0] += np.dot(np.transpose(X), delta_op[num_hls]) * self.learning_rate_init
            bias[0] += np.sum(delta_op[num_hls], axis=0, keepdims=True) * self.learning_rate_init
            # print "shape of weights[0]: ", np.shape(weights[0])
            # print "updated weights: ", weights

            model = {'weights': weights, 'bias': bias, 'ci': count_iters}

            final = last_output
        print "final:", final
        return model

    def convert_to_vector(self, y):

        uniq = np.unique(y)
        print "num labels in y: ",
        vec_size = len(uniq)
        vec = [0] * vec_size
        print vec
        train_y = list()
        for ind in range(len(y)):
            temp = list(vec)
            temp[y[ind]] = 1
            train_y.append(temp)

        # print train_y
        return train_y


    def fit(self, X, y):
        labels = np.unique(y)
        print "labels: ", labels
        train_y = self.convert_to_vector(y)
        # print "---------y------------", y
        num_outputs = len(labels)
        num_inputs = X.shape[1]
        # print "--------num inputs=------------", num_inputs
        self.nn_model = self.build_model(X, train_y, self.hidden_layer_sizes, num_inputs, num_outputs)
        # print self.nn_model
        return self.nn_model

    def compute_result(self, y_vector):
        y_op = list()
        for idx in range(len(y_vector)):
            max_index, max_value = max(enumerate(y_vector[idx]), key=operator.itemgetter(1))
            y_op.append(max_index)
        return y_op


    # Helper function to predict an output (0 or 1)
    def predict(self, X_test):
        wts, bs = self.nn_model['weights'], self.nn_model['bias']
        count_iters = self.ci
        # Forward propagation
        a = []
        layer_input = X_test
        for idx in range(count_iters):
            z = self.sigmoid(np.dot(layer_input, wts[idx]) + bs[idx])
            # a[idx] = logistic.cdf(z[idx])
            a.append(z)
            layer_input = z

        last_output = a[count_iters - 1]
        y_computed = self.compute_result(last_output)

        return y_computed






