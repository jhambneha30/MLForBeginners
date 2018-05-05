import pickle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import random
from matplotlib import pyplot as plt

def loadXY():
	#zippedXY = pickle.load(open("../Vectorizer/zippedXY_wff_1k.p","rb"))
	zippedXY = pickle.load(open("../CNN_features/zippedXY_cnn_wff_1k.p","rb"))
	random.shuffle(zippedXY)
	X,Y = zip(*zippedXY)
	return X,Y



if __name__ == "__main__":

	X,Y = loadXY()
	print "X and Y loaded"
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.80, random_state=0)
	#X_cv,X_test,Y_cv,Y_test = train_test_split(X_ct, Y_ct, test_size=0.50, random_state=0)

	## parameters
	solver = ['svd','lsqr']
	n_components = [1,2,3]
	parameters = {'solver':('svd','lsqr'), 'n_components':[1,2,3]}
	lda_model = LinearDiscriminantAnalysis()
	clf = GridSearchCV(estimator=lda_model, param_grid=parameters,cv=4)
	clf.fit(X_train,Y_train)
	results = clf.cv_results_

	accuracies = results['mean_test_score']
	params = results['params']
	
	xaxis,yaxis,xticks = [],[],[]
	for aa in range(len(params)):
		p = params[aa]
		accuracy = accuracies[aa]
		xaxis.append(len(xticks)); xticks.append(str(p)); yaxis.append(accuracy);

	# creating and saving the plot
	plt.xticks(xaxis, xticks,rotation=90)
	plt.tight_layout()
	plt.plot(xaxis, yaxis)
	#plt.show()
	plt.savefig("lda_cnn_gridsearchCV.png") 
	"""
	predictedY = lda_model.predict(X_test)
	for tt in range(len(Y_test)):
		print "Actual:",Y_test[tt],"  Predicted:",predictedY[tt]
	accuracy = lda_model.score(X_test,Y_test)
	print accuracy
	"""
