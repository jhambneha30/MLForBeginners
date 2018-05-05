import math
import numpy as np
from collections import Counter, defaultdict
import operator


# make sure this class id compatable with sklearn's GaussianNB

class GaussianNB(object):

    def __init__(self):
        # define all the model weights and state here
        self.label_probabilities = dict()
        self.mean_dict = {}
        self.std_dict = {}

    def fit(self, X, Y):
        global labels
        labels = np.unique(Y)
        # print("labels: ", labels)
        x_rows, x_cols = np.shape(X)
        self.label_probabilities = self.compute_probabilities(Y)
        # print("Label probs are:")
        # print(self.label_probabilities)
        separated_set = {}
        for label in labels:
            indices = np.where(Y == label)[0]
            # print("l:", label)
            # print(indices)
            # Picking up the rows from training data whose Y == label and putting into a set called rows_with_label
            rows_with_label = X[np.array(indices), 0:]
            separated_set[label] = rows_with_label
            # print(rows_with_label)
            # print(separated_set[label])

        # print('Separated instances: {0}').format(separated_set)

        # Create a dictionary with class as keys and a list of mean of each attribute value in that class

        for label in labels:
            separate_class = separated_set[label]
            # print("lbl:", label)
            # print("normal: ", separate_class.shape)
            # np.transpose()
            separate_class = separate_class.transpose()
            # print("transposed: ", separate_class.shape)
            ht, wd = separate_class.shape
            temp_mean = np.zeros(ht)
            temp_std = np.zeros(ht)
            index = 0
            for s in separate_class:
                temp_mean[index] += np.mean(s)
                temp_std[index] += np.std(s, ddof=1)
                index += 1
            # for col in np.nditer(separate_class, order='F'):
            #     temp_mean[index] += np.mean(col)
            #     temp_std[index] += np.std(col, ddof=1)
            #
            self.mean_dict[label] = temp_mean
            self.std_dict[label] = temp_std

    @staticmethod
    def compute_probabilities(new_list):
        num_samples = float(len(new_list))
        # Creating a dictionary of counters to store the count of each label. Label is stored as the key in the dict
        target_probs = dict(Counter(new_list))
        for label_key in target_probs.keys():
            target_probs[label_key] /= num_samples
        return target_probs

    @staticmethod
    def gaussian_probability(x, mean, std):
        if std == 0:
            return 1
        exp = math.exp(-(math.pow(x-mean, 2)/(2*math.pow(std, 2))))
        return (1 / (math.sqrt(2 * math.pi) * std)) * exp

    def predict(self, X):
        test_rows, test_cols = np.shape(X)
        print(test_rows, test_cols)
        predictions = np.zeros(test_rows)
        for t in range(0, test_rows):
            test_label_probs = {}

            for l in labels:
                prob_of_label = self.label_probabilities[l]
                likelihood = 0
                for c in xrange(0, test_cols):
                    # print(c)
                    x = X[t][c]
                    mean = self.mean_dict[l][c]
                    stdev = self.std_dict[l][c]
                    res = abs(self.gaussian_probability(x, mean, stdev))
                    # print(res)
                    likelihood += math.log(res) if res > 0.0 else 0
                likelihood += math.log(abs(prob_of_label)) if abs(prob_of_label) > 0.0 else 0
                test_label_probs[l] = likelihood
                # print("l:", l, test_label_probs)
            predictions[t] = max(test_label_probs.iteritems(), key=operator.itemgetter(1))[0]
        # print(predictions)
        return predictions
        # return a numpy array of predictions
