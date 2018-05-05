import pickle
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import random
from matplotlib import pyplot as plt
import numpy as np
import h5py

# def loadXY():
# 	zippedXY = pickle.load(open("../Vectorizer/zippedXY_te.p","rb"))
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

	dataset = "../Vectorizer/XY_wff10.h5"
	X,Y = load_h5py(dataset)

	print "X and Y loaded"	
	#print np.array(X).shape

	#"""
	cnn_X = []
	gap = 2
	sentence = 0
	newf = (len(X[0])/(gap/2)) - 1
	print "newf:",newf
	for xx in X:
		print "Current Sentence : ",sentence
		newfeature = []
		startindex = 0
		for f in range(newf):
			endindex = startindex + gap
			#print "StartIndex :",startindex,"  Endindex:",endindex
			curr_window = list(xx[startindex:endindex]).count(1)
			newfeature.append(curr_window)
			startindex = startindex + gap/2
		cnn_X.append(newfeature)
		sentence += 1
	h5f = h5py.File('XY_wff10_cnn.h5', 'w')
	h5f.create_dataset('X', data=cnn_X)
	h5f.create_dataset('Y', data=Y)
	h5f.close()