from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix
import pickle
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix

def load_h5py(filename):
    with h5py.File(filename, 'r') as hf:
        X = hf['X'][:]
        Y = hf['Y'][:]
    print "X and Y loaded"
    return X, Y

# Use label_binarize to be multi-label like settings
# Y = label_binarize(y, classes=[0, 1, 2])
# n_classes = Y.shape[1]

def compute_precision(y_actual, y_predicted):
	precision = precision_score(y_actual, y_predicted, average="macro")
	return precision

def compute_recall(y_actual, y_predicted):
	recall = recall_score(y_actual, y_predicted, average="macro")
	return recall

def compute_f1_score(y_actual, y_predicted):
	f1 = f1_score(y_actual, y_predicted, average="macro")
	return f1

if __name__ == '__main__':
    #dataset = "../Vectorizer/XY_wff10.h5"
    dataset = "../Feature_reduction/XY_wff10_fs1000.h5"
    #dataset = "../CNN_features/XY_te_cnn.h5" 
    model_file = "../NeuralNet/nn_model_wff10_fs1000.model"
    model = pickle.load(open(model_file,"rb"))
    print "Model loaded"
    X,Y = load_h5py(dataset)
    X = csr_matrix(np.array(X))
    _, X_test, _, Y_test = train_test_split(X,Y, test_size=0.20, random_state=0)
    del X;del Y;
    predicted = model.predict(X_test)
    precision = compute_precision(Y_test,predicted)
    recall = compute_recall(Y_test,predicted)
    f1score = compute_f1_score(Y_test,predicted)
    print "Precision = ",precision
    print "Recall = ",recall
    print "F1 score = ",f1score