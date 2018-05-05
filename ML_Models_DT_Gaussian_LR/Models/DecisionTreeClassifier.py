import numpy as np


# make sure this class id compatable with sklearn's DecisionTreeClassifier

class DecisionTreeClassifier(object):

    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None):
        # define all the model weights and state here
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.col_mean = None
        self.tree_root = None

    def fit(self, X , Y):
        # print("max_depth: ",self.max_depth)
        # print("min samples split:", self.min_samples_split)
        global labels
        labels = np.unique(Y)
        self.col_mean = np.mean(X, axis=0)
        self.tree_root = self.create_dtree(X, Y)

    # We need to split the data basis an attribute. We use the mean value to make the split
    def split_data(self, attribute_index, X, Y):
        less_list_x, more_list_x, less_list_y, more_list_y = list(), list(), list(), list()
        for row_index in range(len(X)):
            if X[row_index][attribute_index] < self.col_mean[attribute_index]:
                less_list_x.append(X[row_index])
                less_list_y.append(Y[row_index])

            else:
                more_list_x.append(X[row_index])
                more_list_y.append(Y[row_index])
        return less_list_x, more_list_x, less_list_y, more_list_y

    # Compute Gini index split data
    def compute_gini(self, lists_x, lists_y):
        # total data points
        total_points = 0
        for list in lists_x:
            total_points += float(len(list))
        # Now, we calculate Gini index for each list caused by the split
        # print("len(lists_x) should be 2: ", len(lists_x))
        gini = 0.0
        for li in range(len(lists_x)):
            list_len = float(len(lists_x[li]))
            if list_len == 0:
                continue
            score = 0.0
            # score the group based on the score for each class
            ##################################
            for label in labels:
                p = float([lists_y[li][row_ind] for row_ind in range(len(lists_y[li]))].count(label)) / float(list_len)
                score += float(p) * float(p)

                # weight the group score by its relative size
            gini += (1.0 - float(score)) * (list_len / float(total_points))
        return gini

    # Select the attribute and the value at which the split should be made.
    # The attribute with
    def best_split(self, X, Y):
        split_index = 100000
        split_val = 10000
        split_gini = 10000
        split_lists_x = None
        split_lists_y = None
        for attr in xrange(len(X[0])):
            for row in X:
                list_x1, list_x2, list_y1, list_y2 = self.split_data(attr, X, Y)
                lists_x = list_x1, list_x2
                lists_y = list_y1, list_y2
                gini_value = self.compute_gini(lists_x, lists_y)
                # print('X%d < %.3f Gini=%.3f' % ((attr + 1), row[attr], gini_value))
                if gini_value < split_gini:
                    split_index, split_val, split_gini, split_lists_x, split_lists_y = attr, row[attr], gini_value, lists_x, lists_y
        # Return node as an object containing the below four things
        return {'split_ind': split_index, 'split_value': split_val, 'split_lists_x': split_lists_x, 'split_lists_y': split_lists_y}

    # Create a leaf using the most frequently occuring label value in the list of rows
    def create_leaf(self, list_x, list_y):
        list_len = len(list_x)
        results = [list_y[row_ind] for row_ind in range(list_len)]
        return max(set(results), key=results.count)

    # Recursive function to create the child splits or to build the leaf nodes
    def child_splits(self, node, depth):
        left_x, right_x = node['split_lists_x']
        left_y, right_y = node['split_lists_y']
        del (node['split_lists_x'])
        del (node['split_lists_y'])
        # if all the rows fall in a single list: either left or right
        if not left_x or not right_x:
            node['left'] = node['right'] = self.create_leaf(left_x + right_x, left_y + right_y)
            return
        # if the tree depth exceeds max_depth specified by user, create leaf node and terminate
        if depth >= self.max_depth:
            node['left'], node['right'] = self.create_leaf(left_x, left_y), self.create_leaf(right_x, right_y)
            return
        # process left child
        if len(left_x) <= self.min_samples_split:
            node['left'] = self.create_leaf(left_x, left_y)
        else:
            node['left'] = self.best_split(left_x, left_y)
            self.child_splits(node['left'], depth + 1)
        # process right child
        if len(right_x) <= self.min_samples_split:
            node['right'] = self.create_leaf(right_x, right_y)
        else:
            node['right'] = self.best_split(right_x, right_y)
            self.child_splits(node['right'], depth + 1)

    def create_dtree(self, X, Y):
        root = self.best_split(X, Y)
        self.child_splits(root, 1)
        return root

    def predict_class(self, node, data_row):
        if data_row[node['split_ind']] < node['split_value']:
            if isinstance(node['left'], dict):
                return self.predict_class(node['left'], data_row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.predict_class(node['left'], data_row)
            else:
                return node['right']

    def predict(self, X):
        predictions = list()
        for row in X:
            prediction_node = self.predict_class(self.tree_root, row)
            predictions.append(prediction_node)
        return predictions
        # return a numpy array of predictions
