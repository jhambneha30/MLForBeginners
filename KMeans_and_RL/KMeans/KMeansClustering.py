import numpy as np
import math
import random
import operator
import matplotlib.pyplot as plt

# make sure this class id compatable with sklearn's LogisticRegression


class KMeansClustering(object):

    def __init__(self, K = 2, max_iters =10, dataset="Plot"):
        # define all the model hyper params here
        self.K = K
        self.max_iters = max_iters
        self.dataset = dataset

    def fit(self, X, Y):
        # Shuffle the data
        zipped = zip(X, Y)
        random.shuffle(zipped)
        X, Y = zip(*zipped)
        X_final = []
        Y_final = []
        cluster_dict_centroids = {}

        # Pick centroids as first k points in X
        for cluster_num in range(self.K):
            cluster_dict_centroids[cluster_num] = X[cluster_num]
        # print "old centroids: ", cluster_dict_centroids

        # Now, we run the loop for max_iters iterations
        for iter in range(self.max_iters):
            cluster_members = {}
            for cluster_num in range(self.K):
                cluster_members[cluster_num] = []

            for index in range(len(X)):
                data_point = X[index]
                # print "data pt:", data_point
                # print "centroid", cluster_dict_centroids[1]
                distances = [np.linalg.norm(data_point - cluster_dict_centroids[centroid]) for centroid in
                                cluster_dict_centroids]
                # print "dist:", distances
                cluster_assigned = distances.index(min(distances))
                cluster_members[cluster_assigned].append(data_point)
                X_final.append(data_point)
                Y_final.append(cluster_assigned)

            # Clusters have been formed now. We need to recompute the centroids as per new clusters.
            for cluster in range(self.K):
                # print "cluster_members[cluster]", cluster_members[cluster]
                # if len(cluster_members[cluster]) > 0:
                cluster_dict_centroids[cluster] = np.mean(cluster_members[cluster], axis=0)
                # else:

            print "---------------Iter no.: -----------------", iter
            # print "new centroids: ", cluster_dict_centroids
        return X_final, Y_final

            # if iter == self.max_iters-1:
            #     plt.scatter(X_final, Y_final, c=y, cmap=plt.cm.get_cmap("jet", 10))
            #     plt.colorbar(ticks=range(10))
            #     plt.clim(-0.5, 9.5)
            #     plt.savefig("Plots/segmentation.png")
            #
            #     plt.show()
            #
            #     plt.close()
            #
            #     plotName = "After KMeans: "
            #     plt.suptitle(plotName + self.dataset, fontsize=20)
            #     # plt.xlabel('user number', fontsize=18)
            #     # plt.ylabel('cluster number', fontsize=16)
            #     plt.savefig('Plots/AfterKMeans_' + self.dataset + '.png')
            #     plt.show()
            #     plt.gcf().clear()

