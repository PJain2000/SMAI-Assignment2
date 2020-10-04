import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn import svm
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from skimage.color import rgb2grey
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD


from sklearn.feature_extraction.text import TfidfVectorizer

class AuthorClassifier:
	classifier = svm.SVC(kernel='rbf', C=12)
	def featurize_text(self, dataframe):
	    vectorizer = TfidfVectorizer()
	    X = vectorizer.fit_transform(dataframe.ravel())
	    
	    svd = TruncatedSVD(n_components=30)
	    Y = svd.fit_transform(X)

	    return Y

	def train(self, filename):
		data = pd.read_csv(filename, delimiter=',')

		author_unique = data['author'].unique()
		data['author'] = data['author'].astype('category')
		data['author'] = data['author'].cat.reorder_categories(author_unique, ordered=True)
		data['author'] = data['author'].cat.codes

		x = data.iloc[:,1:-1]
		y = data.iloc[:,-1]
		y1 = pd.DataFrame(y.values.reshape((y.shape[0],1)))

		feature_list = self.featurize_text(x.values)

		X_train, X_test, y_train, y_test = train_test_split(feature_list, y, test_size=0.3,random_state=0)

		self.classifier.fit(X_train, y_train)

		p = self.classifier.predict(X_test)
		print(accuracy_score(p, y_test))

	def predict(self, filename):
		data = pd.read_csv(filename, delimiter=',')

		author_unique = data['author'].unique()
		data['author'] = data['author'].astype('category')
		data['author'] = data['author'].cat.reorder_categories(author_unique, ordered=True)
		data['author'] = data['author'].cat.codes

		x = data.iloc[:,1:-1]
		y = data.iloc[:,-1]
		y1 = pd.DataFrame(y.values.reshape((y.shape[0],1)))

		feature_list = self.featurize_text(x.values)
		p = self.classifier.predict(feature_list)
		# print(accuracy_score(p, y_test))
		return p



