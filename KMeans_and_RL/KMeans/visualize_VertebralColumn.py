import os
import os.path
import h5py
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

filename = 'vertebral_column_data/column_3C.dat'

# Load the test data
def load_data(filename):
    X = []
    Y = []
    y_dict = {
        'DH': 1,
        'SL': 2,
        'NO': 3,
        'AB': 4
    }
    with open('/home/nehaj/Downloads/Datasets/' + filename) as training_file:
        for line in training_file:
            line = line.split(' ')
            x_temp = line[0:6]
            print x_temp
            x_temp_int = map(float, x_temp)
            Y.append(y_dict[line[6].rstrip()])
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
plt.savefig("Plots/vertebralColumn.png")

plt.show()
plt.close()
