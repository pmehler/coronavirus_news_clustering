
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np
from scipy.spatial import distance
from sklearn.metrics import pairwise_distances_argmin
from scipy.cluster.hierarchy import ward, dendrogram, linkage
from scipy.cluster.hierarchy import cophenet, fcluster
from scipy.spatial.distance import pdist





def hCluster(mat, k, files):
    #'single'
    #'complete'
    #'average'
    Z = linkage(mat, 'ward')
    clusterAssignments = fcluster(Z, k, criterion='maxclust')
    clusterAssignments-=1

    centroids = []
    determineCentroids(centroids, clusterAssignments, k, mat)


    distortionByCluster = []
    calcDistortion(k, distortionByCluster, clusterAssignments, mat, centroids)


    #showTree(Z, files)

    return centroids, clusterAssignments, distortionByCluster


def determineCentroids(centroids, clusterAssignments, numClusters, mat):
    for cI in range(numClusters):
        total = np.zeros(mat[0][:].shape)
        numDocs = 0
        for docID in range(len(clusterAssignments)):
            if (clusterAssignments[docID]==cI):
                numDocs+=1
                total+=mat[docID]
        mean = total / numDocs
        centroids.append(mean)





def calcDistortion(n_clusters, dis_arr, labels, mat, centers):
    # Compute Error
    for i in range(n_clusters):
        n=0
        for j in range(len(mat[labels == i])):
            # sum distance between each vector and respective center
            n += distance.euclidean(mat[labels == i][j],centers[i])**2
        dis_arr.append(n)




#use matplotlib to view dendrogram
def showTree(Z, files):
    i = 0
    for file in files:
        print(i)
        i+=1
        print(file["title"])

    # calculate full dendrogram
    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(
        Z,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
        #truncate_mode='lastp',  # show only the last p merged clusters
        #p=12  # show only the last p merged clusters
    )
    plt.show()
