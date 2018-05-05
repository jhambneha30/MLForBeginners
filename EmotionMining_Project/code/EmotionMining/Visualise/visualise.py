import pickle
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import random
from sklearn.manifold import TSNE
import matplotlib

def loadXY():
	zippedXY = pickle.load(open("../Vectorizer/zippedXY_wff_2k.p","rb"))
	#random.shuffle(zippedXY)
	X,Y = zip(*zippedXY)
	return X,Y

def outliers(X,Y):
	from sklearn.ensemble import IsolationForest
	out =  IsolationForest()
	out.fit(X,Y)
	outliers = list(out.predict(X))
	print "Total outliers : ",outliers



if __name__ == "__main__":

	X,Y = loadXY()
	print "X and Y loaded"
	Ynum = []
	# converting labels to num
	label2num = {}
	label2num["ANGER"],label2num["SADNESS"],label2num["JOY"],label2num["FEAR"],label2num["SURPRISE"] = 0,1,2,3,4

	for yy in range(len(Y)):
		Ynum.append(label2num[Y[yy]])
	print Ynum.index(0)
	print Ynum.index(1)
	print Ynum.index(2)
	print Ynum.index(3)
	print Ynum.index(4)
	"""
	########## 2D PLOT ####################
	# Fitting the tsne with data
	tsne = TSNE(n_components=2, verbose=1) 
	tsne_fit = tsne.fit_transform(X)

	
	# Saving and loading the fitted tsne
	import pickle
	pickle.dump(tsne_fit,open("tsne_fit_wff_2k.p","wb"))
	tsne_fit = pickle.load(open("tsne_fit_wff_2k.p","rb"))
	"""
	"""
	# Visualize the data
	from matplotlib import pyplot as plt
	xx = tsne_fit[:, 0]
	yy = tsne_fit[:, 1]
	colors = ['red','green','blue','black','yellow']
	plt.scatter(xx, yy, c=Ynum, edgecolors='none',cmap=matplotlib.colors.ListedColormap(colors))
	#plt.show()
	

	# Saving the plot in Plots/ folder
	plt.draw()
	plt.savefig("wff_2k_visualise.png")
	#outliers(X,Ynum)
	"""

	################## 3D PLOT #############################
	# Fitting the tsne with data
	tsne = TSNE(n_components=3, verbose=1) 
	tsne_fit = tsne.fit_transform(X)

	
	# Saving and loading the fitted tsne
	import pickle
	pickle.dump(tsne_fit,open("tsne_fit_wff_2k_3d.p","wb"))
	tsne_fit = pickle.load(open("tsne_fit_wff_2k_3d.p","rb"))
	"""
	"""
	# Visualize the data
	from matplotlib import pyplot as plt
	xx = tsne_fit[:, 0]
	yy = tsne_fit[:, 1]
	zz = tsne_fit[:, 2]
	colors = ['red','green','blue','black','yellow']
	from mpl_toolkits.mplot3d import Axes3D
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	print Ynum
	ax.scatter(xx, yy,zz, c=Ynum, edgecolors='none',cmap=matplotlib.colors.ListedColormap(colors))
	#plt.show()
	

	# Saving the plot in Plots/ folder
	plt.draw()
	plt.savefig("wff_2k_visualise_3d__new.png")
	#outliers(X,Ynum)

	
	
	


