import pickle
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import random
import h5py

def featureselectionSKbest(X,Y,feat=1000):
	# feature extraction
	fsm = SelectKBest(chi2, k=feat).fit(X,Y)
	X_new = fsm.transform(X)
	return X_new,Y,fsm

# def loadXY():
# 	zippedXY = pickle.load(open("../Vectorizer/XY_te.h5","rb"))
# 	#zippedXY = pickle.load(open("../Vectorizer/zippedXY_wff_2k.p","rb"))
# 	random.shuffle(zippedXY)
# 	X,Y = zip(*zippedXY)
# 	return X,Y

def load_h5py(filename):
	with h5py.File(filename, 'r') as hf:
		X = hf['X'][:]
		Y = hf['Y'][:]
	print "X and Y loaded"
	return X, Y


if __name__ == "__main__":

	X,Y = load_h5py("../Vectorizer/XY_wff10.h5")
	print "X and Y loaded"
	X_trans,Y_trans,fsm = featureselectionSKbest(X,Y)
	h5f = h5py.File('XY_wff10_fs1000.h5', 'w')
	h5f.create_dataset('X', data=X_trans)
	h5f.create_dataset('Y', data=Y_trans)
	h5f.close()
	print "Feature reduction done."
	