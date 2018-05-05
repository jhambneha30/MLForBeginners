import pickle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import random
from matplotlib import pyplot as plt

def loadXY():
	zippedXY = pickle.load(open("../CNN_features/zippedXY_cnn_wff_1k.p","rb"))
	random.shuffle(zippedXY)
	X,Y = zip(*zippedXY)
	return X,Y



if __name__ == "__main__":

	X,Y = loadXY()
	print "X and Y loaded"
	X_train, X_ct, Y_train, Y_ct = train_test_split(X, Y, test_size=0.60, random_state=0)
	X_cv,X_test,Y_cv,Y_test = train_test_split(X_ct, Y_ct, test_size=0.50, random_state=0)

	## parameters
	solver = ['svd','lsqr']
	n_components = [20,100,1000]
	parameters = [] #{'solver':('svd','lsqr','eigen'), 'tol':[0.0001,0.001,0.01]}
	for sol in solver:
		for n in n_components:
			parameters.append([sol,n])

	xaxis,yaxis,xticks = [],[],[]
	for p in parameters:
		lda_model = LinearDiscriminantAnalysis(solver=p[0],n_components=p[1])
		lda_model.fit(X_train,Y_train)
		accuracy = lda_model.score(X_cv,Y_cv)
		print p, "  Accuracy :",accuracy
		xaxis.append(len(xticks)); xticks.append(str(p)); yaxis.append(accuracy);

	# creating and saving the plot
	plt.xticks(xaxis, xticks,rotation=90)
	plt.tight_layout()
	plt.plot(xaxis, yaxis)
	#plt.show()
	plt.savefig("lda_cnn_grid.png") 
	"""
	predictedY = lda_model.predict(X_test)
	for tt in range(len(Y_test)):
		print "Actual:",Y_test[tt],"  Predicted:",predictedY[tt]
	accuracy = lda_model.score(X_test,Y_test)
	print accuracy
	"""
