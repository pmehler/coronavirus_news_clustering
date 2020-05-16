#https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np
from scipy.spatial import distance
from sklearn.metrics import pairwise_distances_argmin


def find_clusters(X, n_clusters, rseed=2):
    # 1. Randomly choose clusters
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]

    while True:
        # Assign labels based on closest center
        labels = pairwise_distances_argmin(X, centers)

        # Find new centers from means of points
        new_centers = np.array([X[labels == i].mean(0)
                                for i in range(n_clusters)])

        # Check for convergence
        if np.all(centers == new_centers):
            break
        centers = new_centers

    # Compute Error
    dis_arr = np.array([])
    for i in range(n_clusters):
        n=0
        for j in range(len(X[labels == i])):
        	# sum distances within each cluster
            n += distance.euclidean(X[labels == i][j],centers[i])**2
            
        dis_arr = np.append(dis_arr, n)

    return centers, labels, dis_arr
