from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
from operator import itemgetter
import pickle
import h5py
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
import numpy as np


def maximum(matrix):
    i = -1
    return max(enumerate(map(itemgetter(i), matrix)), key=itemgetter(1))

def load_h5py(filename):
    with h5py.File(filename, 'r') as hf:
        X = hf['X'][:]
        Y = hf['Y'][:]
    print "X and Y loaded"
    return X, Y


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = maximum(cm)[1] / 2.
    for i, j in itertools.product(range(len(cm[0])), range(len(cm[1]))):
        plt.text(j, i, format(cm[i][j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i][j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



def plotCM(dataset,model):
    model = pickle.load(open(model,"rb"))
    print "Model loaded"
    X,Y = load_h5py(dataset)
    X = csr_matrix(np.array(X))
    _, X_test, _, Y_test = train_test_split(X,Y, test_size=0.20, random_state=0)
    del X;del Y;
    predicted = model.predict(X_test)
    del X_test
    print "Predicted"
    unique_classes_list = list(set(Y_test))
    print "Ytest=",Y_test
    print "Predicted=",predicted
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(Y_test, predicted, unique_classes_list)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cm=cnf_matrix, classes=unique_classes_list,normalize=True,title='Confusion matrix- TE (FS)')
    plt.savefig('ConfusionMatrix_TE_FS.png')
    plt.show()

if __name__ == '__main__':
    #dataset = "../Vectorizer/XY_wff5.h5"
    dataset = "../Feature_reduction/XY_te_fs500.h5"
    #dataset = "../CNN_features/XY_te_cnn.h5"  
    model = "../NeuralNet/nn_model_te_fs500.model"
    plotCM(dataset,model)