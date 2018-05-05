import pickle
from sklearn import svm
from sklearn.model_selection import train_test_split
import random

def loadXY():
    # zippedXY = pickle.load(open("../Vectorizer/zippedXY.p","rb"))
    zippedXY = pickle.load(open("../Vectorizer/zippedXY_bigrams_5k.p", "rb"))
    random.shuffle(zippedXY)
    X,Y = zip(*zippedXY)
    return X, Y

if __name__ == "__main__":

    X,Y = loadXY()
    print "X and Y loaded"
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.80, random_state=0, max_iter=1000)
    print Y
    svm_model = svm.SVC(random_state=0, kernel='rbf', gamma=2, C=1)
    svm_model.fit(X_train, Y_train)
    # pickle.dump(svm_model, open("svm_bigrams_1k_model", "wb"))
    predictedY = svm_model.predict(X_test)
    for tt in range(len(Y_test)):
        print "Actual:",Y_test[tt],"  Predicted:",predictedY[tt]
    accuracy = svm_model.score(X_test,Y_test)
    print accuracy