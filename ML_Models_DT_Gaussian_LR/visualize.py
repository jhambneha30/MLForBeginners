import os
import os.path
import argparse
import h5py
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("--data", type=str)
parser.add_argument("--plots_save_dir", type=str)

args = parser.parse_args()

# Load the test data
def load_h5py(filename):
	with h5py.File(filename, 'r') as hf:
		X = hf['X'][:]
		Y = hf['Y'][:]
	return X, Y

# X, Y = load_h5py(args.data)
X, Y = load_h5py('/Data/part_A_train.h5')
# print(Y)

labels = list()
for l in Y:
    i = np.where(l == 1)[0][0]
    labels.append(i)

# print(y_list)

tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=300, learning_rate=200)
tsne_results = tsne.fit_transform(X)
x_data = tsne_results[:, 0]
y_data = tsne_results[:, 1]

plt.scatter(x_data, y_data, c=labels, cmap=plt.cm.get_cmap("jet", 10))
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5)
plt.savefig(args.plots_save_dir)

plt.show()

plt.close()

