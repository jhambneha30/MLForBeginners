import numpy as np
import h5py
import idx2numpy
from sklearn.neural_network import MLPClassifier
import pickle
from sklearn.model_selection import train_test_split
import random
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix
import itertools
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
def loadXY():
	#zippedXY = pickle.load(open("../Feature_reduction/zippedXY_wff_fs_2k_1200.p","rb"))
	#zippedXY = pickle.load(open("../CNN_features/zippedXY_cnn_wff_2k_gap2.p","rb"))
	#zippedXY = pickle.load(open("../Vectorizer/zippedXY_wff_2k.p","rb"))
	#zippedXY = pickle.load(open("../Vectorizer/zippedXY_te.p","rb"))
	#zippedXY = pickle.load(open("../CNN_features/zippedXY_cnn_te.p","rb"))
	zippedXY = pickle.load(open("zippedXY_1000.p","rb"))
	random.shuffle(zippedXY)
	X,Y = zip(*zippedXY)
	return X,Y

def create_conf_matrix (conf):
	l = len(conf)
	cm = np.zeros ((l,l))
	ke = conf.keys()
	for i in range(l):
		for j in range (l):
			cm[i][j] = conf [ke[i]][ke[j]]
	return cm
	
def cal_accuracy (A,B, C):
	l = len(A)
	#a is pred
	confusion_matrix = {}		# rows are predicted cols are actual
	for i in C:
		confusion_matrix[i] = {}
		for j in C:
			confusion_matrix[i][j] = 0
		
	ctr = 0 
	for i in range(l) :
		if A[i] == B[i]:
			ctr += 1
		confusion_matrix[A[i]][B[i]] +=1
		
	return confusion_matrix

def plot_confusion_matrix(cm, classes,title,normalize=True,cmap=plt.cm.Blues):

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

	fmt = '.2f' 
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt),horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.savefig( title + "_ConfusionMatrix.png")
	plt.show()

if __name__ == "__main__":

	X,Y = loadXY()
	print "X and Y loaded"
	print "X shape:",np.array(X).shape

	## converting Y to onehot form
	# y = []
	# for yy in Y:
	# 	if yy == "ANGER":
	# 		y.append([1,0,0,0,0])
	# 	elif yy == "FEAR":
	# 		y.append([0,1,0,0,0])
	# 	elif yy == "JOY":
	# 		y.append([0,0,1,0,0])
	# 	elif yy == "SADNESS":
	# 		y.append([0,0,0,1,0])
	# 	elif yy == "SURPRISE":
	# 		y.append([0,0,0,0,1])



	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)
	#print y
	X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.250, random_state=0)
	nn = MLPClassifier(hidden_layer_sizes=(1000 , 500),activation="logistic",solver="adam",max_iter=1000,alpha=0.01,verbose=1)
	nn.fit(X_train,Y_train)
	joblib.dump(nn , "ClassifierDoc2vec.pkl")
	print "model saved"
	#predictedY = nn.predict(X_test)
	#for tt in range(len(Y_test)):
		#print "Actual:",Y_test[tt],"  Prcted:",predictedY[tt]
	#print "Com500,400
	nn = joblib.load ( "ClassifierDoc2vec.pkl")
	accuracy = nn.score(X_val,Y_val)
	print "Validation accuracy : ",accuracy
	accuracy = nn.score(X_train,Y_train)
	print "Training accuracy : ",accuracy
	accuracy = nn.score(X_test,Y_test)
	print "Testing accuracy : ",accuracy
	Y_pred = nn.predict(X_test)
	#print "Testing accuracy : ",accuracy
	#confusion matrix
	classes = ["Anger", "Fear","Joy", "Sadness", "Surprise"]
	confusion_matrix = cal_accuracy (Y_pred, Y_test, [0,1,2,3,4])
	cm1 = create_conf_matrix( confusion_matrix )
	plot_confusion_matrix(cm1,classes, title = "WFF-Using Doc2vec Neural Network")
	print "precision " ,compute_precision (Y_test, Y_pred)
	print "recall " , compute_recall (Y_test, Y_pred)
	print "F1 score ", compute_f1_score (Y_test,Y_pred)
	
