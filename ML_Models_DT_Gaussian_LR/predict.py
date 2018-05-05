import os
import os.path
import argparse
import h5py
import numpy as np
from sklearn.externals import joblib

# python predict.py --model_name LogisticRegression --weights_path Weights/LR_partA --test_data Data/part_A_train.h5
# --output_preds_file predicted_A.txt

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type = str)
parser.add_argument("--weights_path", type = str)
parser.add_argument("--test_data", type = str)
parser.add_argument("--output_preds_file", type = str  )

args = parser.parse_args()


# load the test data
def load_h5py(filename):
    with h5py.File(filename, 'r') as hf:
        X = hf['X'][:]
        Y = hf['Y'][:]
    return X, Y

test_X, test_Y = load_h5py(args.test_data)
if args.model_name == 'GaussianNB':
    # Retrieving the dumped model from the file
    gb = joblib.load(args.weights_path)
    # Making predictions on the new data set using the dumped model
    predicted = gb.predict(test_X)
    # Write a file with the predicted values
    gb_prediction_file = open(args.output_preds_file, 'w')
    for i in predicted:
        gb_prediction_file.write(str(i) + '\n')
elif args.model_name == 'LogisticRegression':
    # Retrieving the dumped model from the file
    lr = joblib.load(args.weights_path)
    # Making predictions on the new data set using the dumped model
    predicted = lr.predict(test_X)
    # Write a file with the predicted values
    lr_prediction_file = open(args.output_preds_file, 'w')
    for i in predicted:
        lr_prediction_file.write(str(i) + '\n')


elif args.model_name == 'DecisionTreeClassifier':
    # load the model
    # Retrieving the dumped model from the file
    dt = joblib.load(args.weights_path)
    # Making predictions on the new data set using the dumped model
    predicted = dt.predict(test_X)
    # Write a file with the predicted values
    dt_prediction_file = open(args.output_preds_file, 'w')
    for i in predicted:
        dt_prediction_file.write(str(i) + '\n')

else:
    raise Exception("Invalid Model name")
