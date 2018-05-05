import pickle
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import random
from matplotlib import pyplot as plt

def loadXY():
	#zippedXY = pickle.load(open("../Vectorizer/zippedXY_wefeelfine_5k.p","rb"))
	zippedXY = pickle.load(open("../CNN_features/zippedXY_cnn_wff_1k.p","rb"))
	random.shuffle(zippedXY)
	X,Y = zip(*zippedXY)
	return X,Y



if __name__ == "__main__":

	X,Y = loadXY()
	print "X and Y loaded"
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.80, random_state=0)
	print Y
	
	penalty = ['l1','l2']
	c_l =  [0.5,1.0,1.5]
	max_iter = [20,50,100]
	res = ['',0,0,0]
	xaxis=[]
	xticks=[]
	yaxis=[]
	for a in penalty:
		for b in c_l:
			for c in max_iter:
				try:
					logistic_model = linear_model.LogisticRegression(verbose=0,penalty=a,C=b,max_iter=c)
					logistic_model.fit(X_train,Y_train)
					predictedY = logistic_model.predict(X_test)
					#for tt in range(len(Y_test)):
						#print "Actual:",Y_test[tt],"  Predicted:",predictedY[tt]
					accuracy = logistic_model.score(X_test,Y_test)
					print "a,b,c:",a,b,c,"  accuracy:",accuracy
					par = []
					par.append(a)
					par.append(b)
					par.append(c)
					xaxis.append(len(xticks));xticks.append(str(par));yaxis.append(accuracy)
					"""
					if(res[3]<=mean_accuracy_lr):
						res[0] = a
						res[1] = b
						res[2] = c
						res[3] = mean_accuracy_lr
					"""
				except ValueError: 
					print "value error comes"
					
	# creating and saving the plot
	plt.xticks(xaxis, xticks,rotation=90)
	plt.plot(xaxis, yaxis)
	plt.tight_layout()
	plt.show()
	plt.savefig("plot2b_cnn_logisticregression.png")
	
	
	
	
	"""
	penalty = ['l2']#,'l2')
	c_l = [0.5]#, 0.5,1.0)
	max_iter = [100]#50,100)
	res = ['',0,0,0]
	xaxis=[]
	xticks=[]
	yaxis=[]
	for a in penalty:
		for b in c_l:
			for c in max_iter:
				try:
					clf = linear_model.LogisticRegression(penalty=a,C=b,max_iter=c)
					#clf = LogisticRegression.LogisticRegression(penalty=a,C=b,max_iter=c)
					print "clf:"
					clf.fit(train_data1x, train_data1y)
					accuracy_train1 = clf.score(test_data1x,test_data1y)
					#print "mean_Accuracy1:",accuracy_train1	
					#print "mean_Accuracy2:",accuracy_train1
					clf = linear_model.LogisticRegression(penalty=a,C=b,max_iter=c)
					clf.fit(train_data2x, train_data2y)
					accuracy_train2= clf.score(test_data2x,test_data2y)
					#print "mean_Accuracy2:",accuracy_train2
					clf = linear_model.LogisticRegression(penalty=a,C=b,max_iter=c)
					clf.fit(train_data3x, train_data3y)
					accuracy_train3= clf.score(test_data3x,test_data3y)
					#print "mean_Accuracy3:",accuracy_train3
					clf = linear_model.LogisticRegression(penalty=a,C=b,max_iter=c)
					clf.fit(train_data4x, train_data4y)
					accuracy_train4= clf.score(test_data4x,test_data4y)
					#print "mean_Accuracy:",accuracy_train4
					clf = linear_model.LogisticRegression(penalty=a,C=b,max_iter=c)
					clf.fit(train_data5x, train_data5y)
					accuracy_train5= clf.score(test_data5x,test_data5y)
					#print "mean_Accuracy:",accuracy_train5
					mean_accuracy_lr = (accuracy_train1+accuracy_train2+accuracy_train3+accuracy_train4+accuracy_train5)/5	
					#print "mean_Accuracy:",mean_accuracy_lr
					print "penalty:",a," C:",b," max_iter:",c," accuracy:",mean_accuracy_lr
					par = []
					par.append(a)
					par.append(b)
					par.append(c)
					xaxis.append(len(xticks));xticks.append(str(par));yaxis.append(mean_accuracy_lr)
					if(res[3]<=mean_accuracy_lr):
						res[0] = a
						res[1] = b
						res[2] = c
						res[3] = mean_accuracy_lr
				except ValueError: 
					print "value error comes"
	print "finsl result: penalty:",res[0]," C:",res[1]," max_iter:",res[2]," accuracy:",res[3]
	# creating and saving the plot
	plt.xticks(xaxis, xticks,rotation=90)
	plt.plot(xaxis, yaxis)
	plt.tight_layout()
	#plt.show()
	plt.savefig(args.plots_save_dir+"\plot2b_logisticregression.png")
	"""


"""
a,b,c: l1 0.5 20   accuracy: 0.86078697422
a,b,c: l1 0.5 50   accuracy: 0.860841248304
a,b,c: l1 0.5 100   accuracy: 0.86078697422
a,b,c: l1 1.0 20   accuracy: 0.865508819539
a,b,c: l1 1.0 50   accuracy: 0.86540027137
a,b,c: l1 1.0 100   accuracy: 0.865671641791
a,b,c: l1 1.5 20   accuracy: 0.862740841248
a,b,c: l1 1.5 50   accuracy: 0.862795115332
a,b,c: l1 1.5 100   accuracy: 0.862957937585
a,b,c: l2 0.5 20   accuracy: 0.837177747626
a,b,c: l2 0.5 50   accuracy: 0.837177747626
a,b,c: l2 0.5 100   accuracy: 0.837177747626
a,b,c: l2 1.0 20   accuracy: 0.844179104478
a,b,c: l2 1.0 50   accuracy: 0.844179104478
a,b,c: l2 1.0 100   accuracy: 0.844179104478
a,b,c: l2 1.5 20   accuracy: 0.845047489824
a,b,c: l2 1.5 50   accuracy: 0.845047489824
a,b,c: l2 1.5 100   accuracy: 0.845047489824
"""
