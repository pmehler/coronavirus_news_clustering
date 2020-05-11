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




############ - TODO, Ideally by Thursday Morning - ##################
#Priority List:
#   write algorithm from scratch
#   calculate distortion in a cluster
#       distortion comparison
#       find optimal k
#   principal component analysis or stemming
#   improve labeling, different methods
#   remove duplicates?
#   More news articles
#   visualize
#       tNe?

stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "said"]

#load files from json
files = []
folderName="covid_news_files"
for filename in listdir(folderName):
    with open(folderName+'/'+ filename) as f:
        try:
            data = json.load(f)
            files.append(data)
        except:
            print(filename, " is corrupt.")

#create array of raw text of article bodies
textbodies=[]
for file in files:
    textbodies.append(file["maintext"])


#preprocess text and choose words/features
cleanText = []
tokenizer = nltk.WordPunctTokenizer()
for doc in textbodies:
    #remove punctuation and make lowercase
    doc = re.sub(r"\p{P}", "", doc)
    doc = doc.lower()
    doc = doc.strip()
    #remove stopwords
    tokens = tokenizer.tokenize(doc)
    filtered_tokens = [token for token in tokens if token not in stopwords]
    doc = ' '.join(filtered_tokens)
    cleanText.append(doc)


#vectorize
tv = TfidfVectorizer(min_df=0., max_df=1., use_idf=True)
tv_matrix = tv.fit_transform(cleanText)
tv_matrix = tv_matrix.toarray()
mat = np.array(tv_matrix) #holds tfidf vals for each word in each document


#run kmeans
numClusters = 10
kmeans = KMeans(n_clusters=numClusters)
kResults = kmeans.fit_predict(mat)#dictinary mapping article text to cluster assignment


#store files organized by cluster in arrays
clusters = [] #array of arrays of files
clusterText = [] #save cleaned text
for i in range(numClusters):
    clusters.append([])
    clusterText.append([])
for i in range(len(kResults)):
    cIndex = kResults[i]
    clusters[cIndex].append(files[i])
    clusterText[cIndex].append(cleanText[i])

# Convert date string to julian date
def datestdtojd (stddate):
    fmt='%Y-%m-%d'
    sdtdate = datetime.strptime(stddate, fmt)
    sdtdate = sdtdate.timetuple()
    jdate = sdtdate.tm_yday
    return(jdate)

#print out the titles in each cluster
average_jdates = []
for cI in range(0,len(clusters)):
    jdate = 0
    print("Cluster ", cI)
    for article in clusters[cI]:
        print(article["title"])
        jdate += datestdtojd(article["date_publish"][0:10])  # we have hh:mm:ss, could be more precise
    average_jdate = jdate/len(clusters[cI]) #what's a better way to represent date of each cluster?
    average_jdates.append(average_jdate)
    print()
    print()
    print()



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
        print("  ", keyword, ": ", counts[keyword])


#print out top 5 occuring keywords per cluster
for cIn in range(0,len(clusterText)):
    print("Cluster "+ str(cIn))
    print(" ", len(clusterText[cIn]), " articles")
    print("Average Julian day of Cluster", average_jdates[cIn])
    print("Keywords:")
    printKeywords(clusterText[cIn])
    print()

