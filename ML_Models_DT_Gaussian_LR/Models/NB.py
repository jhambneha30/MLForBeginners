import numpy as np
from collections import Counter, defaultdict
import operator

# make sure this class id compatable with sklearn's GaussianNB

class GaussianNB(object):

    def __init__(self):
        # define all the model weights and state here
        self.label_probabilities = dict(Counter())
        self.likelihood = {}

    def fit(self, X, Y):
        global labels
        labels = np.unique(Y)
        x_rows, x_cols = np.shape(X)
        # Initializing a defaultdictionary corresponding to every label as key
        for label in labels:
            self.likelihood[label] = defaultdict(list)
        for label in labels:
            indices = np.where(Y == label)[0]
            # Picking up the rows from training data whose Y == label and putting into a set called rows_with_label
            rows_with_label = X[indices, :]
            rows, cols = np.shape(rows_with_label)
            for c in xrange(0, cols):
                self.likelihood[label][c] += list(rows_with_label[:, c])

        for label in labels:
            for c in xrange(0, x_cols):
                self.likelihood[label][c] = self.compute_probs(self.likelihood[label][c])

        self.label_probabilities = self.compute_probs(Y)

    def compute_probs(self, new_list):
        num_samples = float(len(new_list))
        # Creating a dictionary of counters to store the count of each label. Label is stored as the key in the dict
        target_probs = dict(Counter(new_list))
        for label_key in target_probs.keys():
            target_probs[label_key] /= num_samples
        return target_probs

    def predict(self, X):
        test_rows, test_cols  = np.shape(X)
        predictions = np.zeros((test_rows, 1))
        for t in test_rows:
            test_label_probs = {}
            for l in labels:
                prob_of_label = self.label_probabilities[l]
                for c in xrange(0, test_cols):
                    relevant_likelihoods = self.likelihood[l][c]
                    if X[t][c] in relevant_likelihoods.keys():
                        prob_of_label *= relevant_likelihoods[X[t][c]]
                    else:
                        prob_of_label *= 0
                test_label_probs[l] = prob_of_label

            predictions[t] = max(test_label_probs.iteritems(), key=operator.itemgetter(1))[0]

        return predictions
        # return a numpy array of predictions
