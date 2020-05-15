#https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html

#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np
from scipy.spatial import distance
from sklearn.metrics import pairwise_distances_argmin


# r=1000
# c=10
# X = np.random.random((r,c))
# X = X*50

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
            # sum distance between each vector and respective center
            n += distance.euclidean(X[labels == i][j],centers[i])**2
        dis_arr = np.append(dis_arr, n)
        #print("cluster:",i,"Distortion:", distortion[i])

    #print("total distortion:", np.sum(distortion))
    return centers, labels, dis_arr

# centers, labels, distortion = find_clusters(X, 3)
# plt.scatter(X[:, 0], X[:, 1], c=labels,
#             s=50, cmap='viridis');

# dis_arr = np.array([])
# for i in range(1,10):
#     #print(i)
#     centers, labels, distortion = find_clusters(X, i)
#     #print(distortion)
#     dis_arr = np.append(dis_arr, distortion)

# plt.figure()
# plt.plot(dis_arr)
# plt.title('Elbow: Distortion (WCSS) vs. K')
# plt.show()
