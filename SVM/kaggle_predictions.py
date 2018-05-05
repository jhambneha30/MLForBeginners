import os
import os.path

import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier
import csv
from sklearn import svm
from sklearn.metrics import mean_absolute_error
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.multiclass import OneVsRestClassifier

# Function to remove duplicate data points!
def remove_duplicates(arr):
    hash = {}
    for a in arr:
        temp = np.sort(np.array(a["X"]))
        key = ''
        for t in temp:
            key += str(t)
        # key = ''.join(temp)
        if key not in hash:
            hash[key] = {
                "Y": a["Y"],
                "X": a["X"]
            }
    finalArr = []
    for prop in hash:
        finalArr.append(hash[prop])

    # print(finalArr)
    return finalArr


# Load the train/test data
def load_json_train(filename):
    with open(filename) as json_data:
        data = json.load(json_data)
        features_dict_raw = dict()
        print("Complete data size:", len(data))
        data_unique = remove_duplicates(data)
        print("duplicates removed!!!! Size is: ", len(data_unique))
        X = list()
        Y = list()
        for i in range(len(data_unique)):
            X.append(data_unique[i]["X"])
            Y.append(data_unique[i]["Y"])
            for key in X[i]:
                # print(key)
                if key in features_dict_raw:
                    features_dict_raw[key] += 1
                else:
                    features_dict_raw[key] = 1
    # print(features_dict)
    print("feature_dict_raw length", len(features_dict_raw))
    # ======================Freeing memory===================
    del data
    Y = np.array(Y)
    return X, Y, features_dict_raw


# Data points in training data: 472426
# Features in training data: 70134
X, y_array, f_dict = load_json_train('/home/nehaj/Desktop/train.json')
# print(X)
print "--------------------------------------"
# print(Y)

feature_names_list = []
feature_names_dict = {}
feature_vector = []
index = 0
for k in f_dict.keys():
    if f_dict[k] >= 37:
        feature_names_list.append(k)
        feature_vector.append(0)
        feature_names_dict[k] = index
        index += 1

print(len(feature_names_list))
# print("feature names list----------------")
# print(feature_names_list)
# print("feature names dict----------------")
# print(feature_names_dict)

# thefile = open('test.txt', 'w')
vector_list = list()
for x in range(len(X)):
    vector_list.append(list(feature_vector))
    # print("hi:", vector_list[x])
    for k in X[x]:
        if k in feature_names_dict:
            vector_list[x][feature_names_dict[k]] = 1
    # print("bye:", vector_list[x])
    # thefile.write("%s\n" % vector_list[x])

X_array = np.asarray(vector_list, dtype='int32')
# ======================Freeing memory===================
del vector_list
del feature_names_list
del f_dict

print("Done making X_array! size is: ", len(X_array))

# Training linear svm
clf = svm.LinearSVC(C=1, max_iter=1100, random_state=1)
clf.fit(X_array, y_array)

print("SVM trained!")

def load_json_test(filename):
    with open(filename) as json_data:
        data = json.load(json_data)
        features_dict_raw = dict()
        print(len(data))
        X = list()
        for i in range(len(data)):
            X.append(data[i]["X"])
    # print(features_dict)
    # print("==========feature_dict_raw length TEST=======", len(features_dict_raw))
    return X
# ======================Freeing memory===================
del X_array
X_test = load_json_test('/home/nehaj/Desktop/test.json')

vector_list_test = list()
for x in range(len(X_test)):
    vector_list_test.append(list(feature_vector))
    # print("hi:", vector_list[x])
    for k in X_test[x]:
        if k in feature_names_dict:
            vector_list_test[x][feature_names_dict[k]] = 1
    # print("bye:", vector_list[x])
    # thefile.write("%s\n" % vector_list[x])


X_test_array = np.asarray(vector_list_test, dtype='int32')
# ======================Freeing memory===================
print("Test array created!")
del vector_list_test

predicted = clf.predict(X_test_array)
print(predicted)

print("Started writing!")
index = 1
with open('result_kaggle_processed37.csv', 'wb') as out_file:
    csv_writer = csv.writer(out_file)
    csv_writer.writerow(["Id", "Expected"])
    for a in predicted:
        csv_writer.writerow([index, a])
        index += 1

print("Done writing!")








