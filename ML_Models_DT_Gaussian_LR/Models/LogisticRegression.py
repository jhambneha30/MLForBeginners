import numpy as np
import math
import operator

# make sure this class id compatable with sklearn's LogisticRegression


class LogisticRegression(object):

    def __init__(self, penalty='l2', C = 1.0, max_iter=100, verbose=0):
        # define all the model weights and state here
        self.penalty = penalty
        self.max_iter = max_iter
        self.verbose = verbose
        self.C = C
        self.label_parameters_list = {}

    def fit(self, X, Y):
        global labels
        # print("C:", self.C)
        # print("max_iter:", self.max_iter)
        labels = np.unique(Y)

        # While training, we need to compute the theta parameters which can be computed using stochastic GD
        # The method stochasticGD iterates over the data to compute the parameters (when the algo converges)

        for label in labels:
            # print("label: ", label)
            Y_copy = np.copy(Y)
            # print("Y_copy original:", Y_copy)
            indices = np.where(Y == label)[0]
            mask = np.ones(Y_copy.shape, dtype=bool)
            mask[indices] = False
            Y_copy[~mask] = 1
            Y_copy[mask] = 0
            # print("Y_copy changed:", Y_copy)
            self.label_parameters_list[label] = self.stochasticGD(X, Y_copy)

    # The below function computes the parameters using the stochastic Gradient Descent method
    def stochasticGD(self, X, Y):
        params = [0.0 for i in range(len(X[0]))]
        for it in range(self.max_iter):
            index = 0
            for row in X:
                predicted_y = self.compute(row, params)
                error = Y[index] - predicted_y
                index += 1
                params[0] = float(params[0]) + self.C * error * predicted_y * (1.0 - predicted_y)
                for i in range(len(row)):
                    params[i] = float(params[i]) + self.C * error * predicted_y * (1.0 - predicted_y) * row[i-1]
        return params

    # Computing the value (prediction) using the parameters
    def compute(self, vector, param_list):
        res = param_list[0]
        for i in range(len(vector)):
            res += float(param_list[i] * vector[i-1])
        try:
            final_res = 1.0 / (1.0 + math.e**(-res))
        except OverflowError:
            final_res = 0.5
        return final_res

    def predict(self, X):
        ln = len(X)
        predictions = np.zeros(ln)
        for t in xrange(0, ln):
            test_label_probs = {}
            for l in labels:
                # print("params list", self.label_parameters_list[l])
                prob_of_label = float(self.compute(X[t], self.label_parameters_list[l]))
                test_label_probs[l] = prob_of_label
            # print(X[t])
            # print(test_label_probs)
            predictions[t] = max(test_label_probs.iteritems(), key=operator.itemgetter(1))[0]
            # print(X[t])
            # print(predictions[t])
        # return a numpy array of predictions
        return predictions
