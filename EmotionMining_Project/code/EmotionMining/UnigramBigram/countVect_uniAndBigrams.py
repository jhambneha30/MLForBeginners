import pickle
import random

def loadXY():
    X = []
    zippedXY_uni = pickle.load(open("../Vectorizer/zippedXY_unigrams_10k.p", "rb"))
    zippedXY_bi = pickle.load(open("../Vectorizer/zippedXY_bigrams_10k_lessFeatures.p", "rb"))
    X_uni, Y = zip(*zippedXY_uni)
    X_bi, Y_bi = zip(*zippedXY_bi)
    print "Length of X_uni", len(X_uni)
    print "Length of X_bi", len(X_bi)
    print Y
    print Y_bi
    for ind in range(len(X_uni)):
        X.append(X_uni[ind] + X_bi[ind])

    print "length of combined: ", len(X)

    return X, Y


X_list, Y_list = loadXY()
zippedXY = zip(X_list, Y_list)
pickle.dump(zippedXY, open("zippedXY_uni_bi_10k_lessFeatures.p","wb"))


