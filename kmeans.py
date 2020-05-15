import json
from os import listdir
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.cluster import KMeans
import nltk
import regex as re
from sklearn.feature_extraction.text import CountVectorizer
from datetime import datetime
from kmeans_scratch import find_clusters
from hierarchical import hCluster
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
from scipy.spatial import distance
from sklearn.metrics import pairwise_distances_argmin
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA



############ - TODO, Ideally by Thursday Morning - ##################
#Priority List:
# merge
# compare clustering methods

stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "said"]

#kmeans, hierarchical, comparison
mode = "hierarchical"
folderName="testfiles"
numClusters = 3 #12


def loadFiles(foldername, files):
    '''
    Load files from foldername into files array
    '''
    for filename in listdir(foldername):
        with open(foldername+'/'+ filename) as f:
            try:
                data = json.load(f)
                files.append(data)
            except:
                print(filename, " is corrupt.")


def preprocessText(files, cleanTextBodies):
    '''
    Put array of raw text of article bodies into cleanTextArr
    clean text by removing stopwords
    '''
    #preprocess text and choose words/features
    tokenizer = nltk.WordPunctTokenizer()
    for file in files:
        doc = file["maintext"]
        #remove punctuation and make lowercase
        doc = re.sub(r"\p{P}", "", doc)
        doc = doc.lower()
        doc = doc.strip()
        #remove stopwords
        tokens = tokenizer.tokenize(doc)
        filtered_tokens = [token for token in tokens if token not in stopwords]
        doc = ' '.join(filtered_tokens)
        cleanTextBodies.append(doc)


def vectorize(cleanText):
    '''
    vectorize the words
    '''
    tv = TfidfVectorizer(min_df=0., max_df=1., use_idf=True, max_features=8192)
    tv_matrix = tv.fit_transform(cleanText)
    tv_matrix = tv_matrix.toarray()
    mat = np.array(tv_matrix) #holds tfidf vals for each word in each document
    return mat


def runKMeans(mat, numClusters):
    '''
    '''
    return find_clusters(mat, numClusters)


def runHierarchical(mat, numClusters, files):
    '''
    '''
    print('running h')
    return hCluster(mat, numClusters, files)



# Convert date string to julian date
def datestdtojd (stddate):
    fmt='%Y-%m-%d'
    sdtdate = datetime.strptime(stddate, fmt)
    sdtdate = sdtdate.timetuple()
    jdate = sdtdate.tm_yday
    return(jdate)



#print most frequently occurring words in each cluster:
#currently uses CountVectorizer
def printKeywords(arrOfText):
    #currently this transforms the array into one long string
    #TODO change to looping through different strings and summing count
    c1text = " ".join(arrOfText)

    #use CountVectorizer to count word occurences
    vectorizer = CountVectorizer()
    vectorizer.fit([c1text]) #double check behavior
    #dictionary of words mapping to their index in vector
    countDict = vectorizer.vocabulary_
    #vector of word counts
    vector = vectorizer.transform([c1text])

    #dictionary containing {word:# word occurences}
    counts = {}
    for word in countDict:
        counts[word] = vector[0, countDict[word]]

    #create list of words sorted by frequency
    sortedCount = sorted(counts, key = counts.get, reverse=True)

    #take top 5 keywords and print
    keywords = []
    for wordi in sortedCount:
        if(len(keywords)<5):
            keywords.append(wordi)
    for keyword in keywords:
        print("    ", keyword, ": ", counts[keyword])













def outputResults(results, distortion, centers):
    '''
    '''
    #partition files by assigned cluster
    clusters = [] #array of array of files
    clusterText = [] #save cleaned text
    for i in range(numClusters):
        clusters.append([])
        clusterText.append([])
    for i in range(len(results)):
        cIndex = results[i]
        clusters[cIndex].append(files[i])
        clusterText[cIndex].append(cleanText[i])
    print("total distortion is ", np.sum(distortion))
    #print out the titles in each cluster
    titles = []
    average_jdates = []
    for cI in range(0,len(clusters)):
        jdate = 0
        min_dist=10000
        index = -1
        print("Cluster ", cI)
        for article in clusters[cI]:
            print(article["title"])
            jdate += datestdtojd(article["date_publish"][0:10])  # we have hh:mm:ss, could be more precise
        average_jdate = jdate/len(clusters[cI]) #what's a better way to represent date of each cluster?
        average_jdates.append(average_jdate)

        #For each element in mat which is a label in kResults, find the min distance to center of that label
        for j in range(len(mat[kResults==i])):
            if min_dist > distance.euclidean(mat[kResults==i][j],centers[i]):
                min_dist = distance.euclidean(mat[kResults==i][j],centers[i])
                index = j
        titles.append(index)
        print()
        print()
        print()

    #print out top 5 occuring keywords per cluster
    for cIn in range(0,len(clusterText)):
        print("Cluster "+ str(cIn))
        print("  cluster distortion is ", distortion[cIn])
        print("  averaged cluster distortion is ", distortion[cIn]/len(clusters[cIn]))
        print(" ", len(clusterText[cIn]), " articles")
        print("  Average Julian day", average_jdates[cIn])
        print("  Title closest to center:")
        print("    ", clusters[cIn][titles[cIn]]["title"])
        print("  Keywords:")
        printKeywords(clusterText[cIn])
        print()





##################################### check for args here

files = []
loadFiles(folderName, files)

cleanText = []
preprocessText(files, cleanText)

mat = vectorize(cleanText)

if mode == "kmeans":
    centers, kResults, distortion = runKMeans(mat, numClusters)
    outputResults(kResults, distortion, centers)
elif mode == "hierarchical":
    centers, kResults, distortion = runHierarchical(mat, numClusters, files)
    outputResults(kResults, distortion, centers)
else: #default both
    #centers, results, distortion = runKMeans(mat, numClusters)
    results = runHierarchical(mat, numClusters)
    #output comparison










########################## kmeans elbow #########################

# dis_arr = np.array([])
# for numClusters in range(1,25):
#     print(numClusters, "of 25")
#     centers, kResults, distortion = find_clusters(mat, numClusters)
#     #print(distortion)
#     dis_arr = np.append(dis_arr, distortion)

# plt.figure()
# plt.plot(dis_arr)
# plt.title('Elbow: Distortion (WCSS) vs. K')
# plt.show()

#################################################################



############################## PLOTTING #####################################
# pca = PCA(n_components=0.95, random_state=42)
# X_reduced= pca.fit_transform(mat)
# kmeans = KMeans(n_clusters=numClusters)
# y_pred = kmeans.fit_predict(X_reduced)#dictinary mapping article text to cluster assignment
# tsne = TSNE(verbose=1, perplexity=100, random_state=42)
# X_embedded = tsne.fit_transform(mat)

# # sns settings
# sns.set(rc={'figure.figsize':(15,15)})

# # colors
# palette = sns.hls_palette(12, l=.4, s=.9)

# # plot
# sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y_pred, legend='full', palette=palette)
# plt.title('t-SNE with Kmeans Labels')
# plt.savefig("improved_cluster_tsne.png")
# plt.show()
