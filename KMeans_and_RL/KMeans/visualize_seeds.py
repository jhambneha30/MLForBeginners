import os
import os.path
import h5py
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

filename = 'seeds_dataset.txt'

# Load the test data
def load_data(filename):
    X = []
    Y = []
    with open('/home/nehaj/Downloads/Datasets/' + filename) as training_file:
        for line in training_file:
            line = line.split('\t')
            x_temp = line[0:7]
            print x_temp
            x_temp_int = map(float, x_temp)
            Y.append(int(line[7]))
            X.append(x_temp_int)

        print "X at index 0", X[0]
        print "Y at index 0", Y[0]
    return X, Y


X, y = load_data(filename)
# print(Y)


tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=300, learning_rate=200)
tsne_results = tsne.fit_transform(X)
x_data = tsne_results[:, 0]
y_data = tsne_results[:, 1]

plt.scatter(x_data, y_data, c=y, cmap=plt.cm.get_cmap("jet", 10))
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5)
plt.savefig("Plots/seeds.png")

plt.show()

plt.close()
