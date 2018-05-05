import numpy as np
import h5py
import idx2numpy
from sklearn.neural_network import MLPClassifier
import pickle
from sklearn.model_selection import train_test_split
import random

	
# def loadXY():
# 	#zippedXY = pickle.load(open("../Feature_reduction/zippedXY_wff_fs_2k_1200.p","rb"))
# 	#zippedXY = pickle.load(open("../CNN_features/zippedXY_cnn_wff_2k_gap2.p","rb"))
# 	zippedXY = pickle.load(open("../Vectorizer/zippedXY_wff_10k.p","rb"))
# 	#zippedXY = pickle.load(open("../Vectorizer/zippedXY_te.p","rb"))
# 	#zippedXY = pickle.load(open("../CNN_features/zippedXY_cnn_te.p","rb"))
# 	#zippedXY = pickle.load(open("../Feature_reduction/zippedXY_te_fs_1000.p","rb"))
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

	X,Y = load_h5py("../CNN_features/XY_te_cnn.h5")
	#X,Y = load_h5py("../Feature_reduction/XY_wff5_fs1000.h5")
	#X,Y = load_h5py("../Vectorizer/XY_te.h5")
	#X,Y = load_h5py("../Vectorizer/XY_wff10.h5")
	print "X and Y loaded"
	#print "X shape:",np.array(X).shape

	X_train_val, _, Y_train_val, _ = train_test_split(X,Y, test_size=0.20, random_state=0)
	del X;del Y
	X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=0.25, random_state=0)
	del X_train_val;del Y_train_val
	#print y
	
	nn = MLPClassifier(hidden_layer_sizes=(200),activation="relu",solver="adam",max_iter=30,alpha=0.1,verbose=1)
	nn.fit(X_train,Y_train)
	pickle.dump(nn,open("nn_model_te_cnn.model","wb"))
	#nn = pickle.load(open("nn_model_te.model","rb"))
	#redictedY = nn.predict(X_val)
	#for tt in range(len(Y_test)):
		#print "Actual:",Y_test[tt],"  Prcted:",predictedY[tt]
	#print "Com500,400
	#print "Com500,400
	train_accuracy = nn.score(X_train,Y_train)
	print "Training accuracy : ",train_accuracy
	val_accuracy = nn.score(X_val,Y_val)
	print "Validation accuracy : ",val_accuracy
	test_accuracy = nn.score(X_test,Y_test)
	print "Validation accuracy : ",test_accuracy


## 100, 20 iterations
# Validation accuracy :  0.95152995153
# Validation accuracy :  0.885780885781

## 100, 30 iterations
# Training accuracy :  0.958115958116
# Validation accuracy :  0.879675879676

## 200, 30 iterations
# Training accuracy :  0.958152958153
# Validation accuracy :  0.878121878122

############# TE #################
## 50, 1000 iterations
# Training accuracy :  0.987219209915
# Validation accuracy :  0.344947735192

## 50, 1000 iterations, alpha= 0.1
# Training accuracy :  0.615027110767
# Validation accuracy :  0.389082462253

## 50, 1000 iterations, alpha= 1
# Training accuracy :  0.431835786212
# Validation accuracy :  0.376306620209

## 100, 1000 iterations
# Training accuracy :  0.656274206042
# Validation accuracy :  0.373403019744

## 200, 1000 iterations
# Training accuracy :  0.64446165763
# Validation accuracy :  0.371660859466

