import os
from random import shuffle
import idx2numpy
import h5py
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np


# Load the test data
def load_h5py(filename):
	with h5py.File(filename, 'r') as hf:
		X = hf['X'][:]
		Y = hf['Y'][:]
	return X, Y

# X, Y = load_h5py(args.data)
# X, Y = load_h5py('dataset_partA.h5')
# print X
# print Y

def load_data(filename):
	if ".h5" in filename[0]:
		X,Y = load_h5py(filename[0])
	else:
		X = idx2numpy.convert_from_file(filename[0])
		Y = idx2numpy.convert_from_file(filename[1])
	return X,Y

def loadAndProcesstXY(filename):
	X,Y = load_data(filename)
	new_X,new_Y = [],[]
	for xx in range(len(X)):
		curr_x = np.ndarray.flatten(X[xx])
		curr_x_f = curr_x.astype(np.float)
		curr_x_n = np.divide(curr_x_f,255)
		curr_x_n_short = np.around(curr_x_n,decimals = 3)
		#print curr_x_n_short
		new_X.append(curr_x_n_short)
		new_Y.append([Y[xx]])
	X = np.array(new_X)#np.divide(np.array(new_X),255)
	# Y = np.array(new_Y)
	print "Shape of X:",X.shape
	print "Shape of Y:",Y.shape
	return X,Y


# Load test data
# X, y = load_data('dataset_partA.h5')
X, y = loadAndProcesstXY(["mnist/train-images.idx3-ubyte", "mnist/train-labels.idx1-ubyte"])
# combined = list(zip(X, y))
# shuffle(combined)

# X_small[:], y_small[:] = zip(*combined)
X_small = list(X[0:12000])
y_small = list(y[0:12000])
# Converting 3D array X to 2D array X
# nsamples, nx, ny = X.shape
# X = X.reshape((nsamples, nx*ny))

# labels = list()
# for l in Y:
#     i = np.where(l == 1)[0][0]
#     labels.append(i)

# print(y_list)

tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=300, learning_rate=200)
tsne_results = tsne.fit_transform(X_small)
x_data = tsne_results[:, 0]
y_data = tsne_results[:, 1]

plt.scatter(x_data, y_data, c=y_small, cmap=plt.cm.get_cmap("jet", 10))
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5)
plt.savefig('Plots/MNIST.png')

plt.show()

plt.close()

