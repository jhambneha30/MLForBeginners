import numpy as np
import math
import random
import operator
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_mutual_info_score

# make sure this class id compatable with sklearn's LogisticRegression


class KMeansClustering(object):

    def __init__(self, K = 2, max_iters =100, dataset_name="default"):
        # define all the model hyper params here
        self.K = K
        self.max_iters = max_iters
        self.dataset_name = dataset_name
        self.Y_predicted = []
        self.X_new = None
        self.Y_true = None

    def has_converged(self, old_list, new_list):
        return set([tuple(a) for a in new_list]) == set([tuple(a) for a in old_list])

    def fit(self, X, Y):
        # Shuffle the data
        zipped = zip(X, Y)
        random.shuffle(zipped)
        X, Y = zip(*zipped)
        self.Y_true = list(Y)

        cluster_dict_centroids = {}
        cluster_centroids_list = []

        # Pick centroids as first k points in X
        cluster_centroids_old_list = []
        for cluster_num in range(self.K):
            cluster_dict_centroids[cluster_num] = X[cluster_num]
            cluster_centroids_list.append(X[cluster_num])
            cluster_centroids_old_list.append(X[cluster_num + 2])
        # print "old centroids: ", cluster_dict_centroids

        # Now, we run the loop for max_iters iterations
        iterations = 0
        # check_convergence_old = -100
        # check_convergence_new = -200
        graph_x_dt, graph_y_dt = list(), list()
        while not self.has_converged(cluster_centroids_old_list, cluster_centroids_list):
            cluster_members = {}
            X_final = []
            Y_final = []

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
            cluster_centroids_old_list = list(cluster_centroids_list)


            sum_dist = 0.0
            for c in cluster_dict_centroids:
                for member in cluster_members[c]:
                    d = np.linalg.norm(cluster_dict_centroids[c] - member)
                    sum_dist += d
            print "---------------Iteration no.: -----------------", iterations
            print sum_dist
            self.Y_predicted = list(Y_final)
            self.X_new = list(X_final)
            graph_x_dt.append(iterations)
            graph_y_dt.append(sum_dist)
            # cluster_dict_centroids_old.clear()
            # cluster_dict_centroids_old.update(cluster_dict_centroids)
            # cluster_dict_centroids_old = cluster_dict_centroids
            # Clusters have been formed now. We need to recompute the centroids as per new clusters.
            for cluster in range(self.K):
                # print "cluster_members[cluster]", cluster_members[cluster]
                mean = np.mean(cluster_members[cluster], axis=0)
                cluster_dict_centroids[cluster] = mean
                cluster_centroids_list[cluster] = mean

            iterations += 1
            if iterations == self.max_iters:
                break
            # print "new centroids: ", cluster_dict_centroids
            # creating and saving the plot
        # plt.xticks(graph_x_dt, param_comb_list_dt, rotation=90)
        plt.plot(graph_x_dt, graph_y_dt)
        plt.xlabel('Iterations', fontsize=14)
        plt.ylabel('Objective Function', fontsize=14)
        plotName = "KMeans-Objective vs Iteration: " + self.dataset_name
        plt.title(plotName, fontsize=12)
        plt.tight_layout()
        # plt.show()
        plt.savefig("Plots/ObjectiveFuncVSIterations_"+self.dataset_name+".png")
        plt.close()


        return self.X_new, self.Y_predicted

    def metrics(self):
        ari = adjusted_rand_score(self.Y_true, self.Y_predicted)
        nmi = normalized_mutual_info_score(self.Y_true, self.Y_predicted)
        ami = adjusted_mutual_info_score(self.Y_true, self.Y_predicted)
        metric_values ={"ari": ari, "nmi": nmi, "ami": ami}
        return metric_values



