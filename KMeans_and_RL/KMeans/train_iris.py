import matplotlib.pyplot as plt
import numpy as np
# from KMeansClustering import KMeansClustering
from KM import KMeansClustering
from sklearn.manifold import TSNE
import matplotlib.cm as cm

filename = 'Iris/iris.data.txt'

# Load the test data
def load_data(filename):
    X = []
    Y = []
    with open('/home/nehaj/Downloads/Datasets/' + filename) as training_file:
        for line in training_file:
            line = line.split(',')
            x_temp = line[0:4]
            # print x_temp
            x_temp_int = map(float, x_temp)
            Y.append(line[4])
            X.append(x_temp_int)

        print "X at index 0", X[0]
        print "Y at index 0", Y[0]
    X = np.array(X)
    Y = np.array(Y)
    return X, Y


X, y = load_data(filename)
num_labels = len(set(y))
# print(Y)
avg_ari = 0.0
avg_nmi = 0.0
avg_ami = 0.0

for i in range(5):
    km = KMeansClustering(K=num_labels, max_iters=15, dataset_name="iris")
    X_final, y_final = km.fit(X, y)

    print "y_final=============", y_final
    print "Iris"
    metric_vals = km.metrics()
    avg_ari += metric_vals["ari"]
    avg_nmi += metric_vals["nmi"]
    avg_ami += metric_vals["ami"]
    print "ari: ", metric_vals["ari"]
    print "nmi: ", metric_vals["nmi"]
    print "ami: ", metric_vals["ami"]

avg_ari /= 5
avg_nmi /= 5
avg_ami /= 5
print "Average ari: ", avg_ari
print "Average nmi: ", avg_nmi
print "Average ami: ", avg_ami

# Plot the new scatter plot after k means
# tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=300, learning_rate=200)
# tsne_results = tsne.fit_transform(X_final)
# x_data = tsne_results[:, 0]
# y_data = tsne_results[:, 1]
#
# # y_colorcodes = [x+x for x in y_final]
#
# plt.scatter(x_data, y_data, c=y_final, cmap=plt.cm.get_cmap("jet", 10))
# plotName = "After KMeans: Iris"
# plt.suptitle(plotName, fontsize=20)
# plt.colorbar(ticks=range(10))
# plt.clim(-0.5, 9.5)
# # plt.xlabel('user number', fontsize=18)
# # plt.ylabel('cluster number', fontsize=16)
# plt.savefig("Plots/AfterKMeans-iris.png")
# plt.show()
# plt.gcf().clear()




