import pandas as pd
from random import sample
from math import sqrt
from numpy import mean
import numpy as np
import copy
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

class Cluster:
	k = 5
	MAX_ITER = 500
	def dist(self, v1, v2):
	    return np.sqrt(np.sum((np.array(v1) - np.array(v2)) ** 2)) 

	def initializeCenters(self, df, k):
	    random_indices = sample(range(len(df)), k)
	    centers = [list(df.iloc[idx]) for idx in random_indices]
	    return centers

	def computeCenter(self, df, k, cluster_labels):
	    cluster_centers = list()
	    data_points = list()
	    for i in range(k):
	        for idx, val in enumerate(cluster_labels):
	            if val == i:
	                data_points.append(list(df.iloc[idx]))
	        cluster_centers.append(list(map(mean, zip(*data_points))))
	    return cluster_centers

	def assignCluster(self, df, k, cluster_centers):
	    cluster_assigned = list()
	    for i in range(len(df)):
	        distances = [self.dist(list(df.iloc[i]),center) for center in cluster_centers]
	        min_dist, idx = min((val, idx) for (idx, val) in enumerate(distances))
	        cluster_assigned.append(idx)
	    return cluster_assigned

	def kmeans(self, df, k, class_labels):
	    cluster_centers = self.initializeCenters(df, k)
	    curr = 1

	    while curr <= self.MAX_ITER:
	        cluster_labels = self.assignCluster(df, k, cluster_centers)
	        cluster_centers = self.computeCenter(df, k, cluster_labels)
	        curr += 1

	    return cluster_labels, cluster_centers

	def featurize_text(self, data):
	    vectorizer = TfidfVectorizer()
	    X = vectorizer.fit_transform(data.ravel())
	    
	    svd = TruncatedSVD(n_components=2)
	    Y = svd.fit_transform(X)
	    Z = StandardScaler().fit_transform(Y)

	    return Z

	def cluster(self, filename):
		files = os.listdir(filename)

		x = []
		y_labels = []
		for file in files:
		    y_labels.append(int(file.split("_",1)[1][0]))
		    f = open(filename+file, 'r', errors='replace')
		    data = f.read()
		    x.append(data)
		    f.close()

		z = self.featurize_text(np.array(x))

		df = pd.DataFrame(z) 
		class_labels = ['business', 'entertainment', 'politics', 'sport', 'tech']

		labels, centers = self.kmeans(df, self.k, class_labels)

		return labels




