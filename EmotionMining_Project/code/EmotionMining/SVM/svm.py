import pickle
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import random
import h5py

def load_h5py(filename):
    with h5py.File(filename, 'r') as hf:
        X = hf['X'][:]
        Y = hf['Y'][:]
    print "X and Y loaded"
    return X, Y


def loadXY():
	#zippedXY = pickle.load(open("../Vectorizer/zippedXY_wff_2k.p","rb"))
	#zippedXY = pickle.load(open("../CNN_features/zippedXY_cnn_wff_2k_gap4.p","rb"))
	#zippedXY = pickle.load(open("../Feature_reduction/zippedXY_wff_fs_2k.p","rb"))
	#zippedXY = pickle.load(open("../Vectorizer/zippedXY_te.p","rb"))
	#zippedXY = pickle.load(open("../CNN_features/zippedXY_cnn_te.p","rb"))
	#zippedXY = pickle.load(open("../Feature_reduction/zippedXY_te_fs.p","rb"))
	random.shuffle(zippedXY)
	X,Y = zip(*zippedXY)
	return X,Y



if __name__ == "__main__":

	#X,Y = loadXY()
	dataset = "../Vectorizer/XY_wff10.h5"
	X,Y = load_h5py(dataset)
	print "X shape:",len(X),"   ",len(X[0])
	print "X and Y loaded"
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)
	del X;del Y;
	#print Y
	svm_model = LinearSVC(random_state=0,verbose=1)
	svm_model.fit(X_train,Y_train)
	pickle.dump(svm_model,open("svm_model.p","wb"))
	predictedY = svm_model.predict(X_test)
	for tt in range(len(Y_test)):
		print "Actual:",Y_test[tt],"  Predicted:",predictedY[tt]
	accuracy = svm_model.score(X_test,Y_test)
	print accuracy
