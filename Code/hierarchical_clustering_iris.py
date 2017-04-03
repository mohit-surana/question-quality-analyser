# Invocation: python3 hierarchical_clustering_iris.py

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
import sklearn.metrics as sm

import pandas as pd
import numpy as np

# Only needed if you want to display your plots inline if using Notebook
# change inline to auto if you have Spyder installed
#%matplotlib inline
# import some data to play with

printed_level = {}
# Set the size of the plot

def getClusters(no_of_clusters, x, nameOfClass, level, pos):
    # Create a colormap

    colormap = np.array(['red', 'lime', 'black', '#%02x%02x%02x' % (0, 255,0)
                        , '#%02x%02x%02x' % (0, 0,255), '#%02x%02x%02x' % (100, 0, 100)
                        , '#%02x%02x%02x' % (0, 100,100), '#%02x%02x%02x' % (100, 100,0)
                        , '#%02x%02x%02x' % (150, 100,50), '#%02x%02x%02x' % (50, 100,150)])

    # K Means Cluster
    model = KMeans(n_clusters=no_of_clusters)
    model.fit(x)
    # This is what KMeans thought

    predY = np.choose(model.labels_, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).astype(np.int64)
    #print('predY:', predY)

    # View the results
    # Plot Predicted with corrected values
    if(level not in printed_level):
        printed_level[level] = True
        plt.figure(figsize=(10,5))
        plt.scatter(x.Petal_Length, x.Petal_Width, c=colormap[predY], s=40)
        plt.title('level:' + str(level) + ' ' + nameOfClass + pos)
    # Performance Metrics
    #print('Accuracy:', sm.accuracy_score(y, predY))
    # Confusion Matrix
    #print('Confusion Matrix:', sm.confusion_matrix(y, predY))
    return predY

def recurseHier(no_of_clusters, data, x, labelString, level, pos):
    tempDict = dict()
    if(no_of_clusters == 1):
        return
    class_clusters = dict()
    data = data.tolist()
    clusters = getClusters(min(no_of_clusters, len(x)), x, labelString, level+1, pos)
    for ith_cluster in range(len(clusters)):
        if clusters[ith_cluster] in class_clusters.keys():
            class_clusters[clusters[ith_cluster]].append(data[ith_cluster])
        else:
            class_clusters[clusters[ith_cluster]] = [data[ith_cluster]]

    tempDict[pos] = class_clusters
    if(level in globalDict.keys()):
        globalDict[level].append(tempDict)
    else:
        globalDict[level] = list()
        globalDict[level].append(tempDict)
    for i in class_clusters.keys():
        subarray = np.array(class_clusters[i])
        subarray1 = pd.DataFrame(subarray)
        subarray1.columns = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']
        if(len(class_clusters) > 1):
            recurseHier(min(no_of_clusters, len(subarray)), subarray, subarray1, labelString, level+1, str(i+1))

globalDict = dict()
if __name__ == "__main__":
    iris = datasets.load_iris()
    # Store the inputs as a Pandas Dataframe and set the column names
    x = pd.DataFrame(iris.data)
    x.columns = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']
    y = pd.DataFrame(iris.target)
    y.columns = ['Targets']
    no_of_clusters = 2
    recurseHier(no_of_clusters, iris.data, x, 'Figure', 1, '1')

    print('globalDict')
    print()
    for key, value in globalDict.items():
        print(key, ':', value)
        print()
    plt.show()
