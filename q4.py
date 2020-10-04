import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error

class Weather:
	theta1 = []
	def  cal_cost_mse(self,theta,X,y):
	    m = len(y)
	    predictions = X.dot(theta)
	    cost = (1/2*m) * np.sum(np.square(predictions-y))
	    return cost

	def  cal_cost_mae(self,theta,X,y):
	    predictions = X.dot(theta)
	    cost = mean_absolute_error(y, predictions)
	    return cost

	def  cal_cost_mape(self,theta,X,y):
	    predictions = X.dot(theta)
	    cost =  np.mean(np.abs((y - predictions) / y)) * 100
	    return cost

	def gradient_descent(X,y,theta,learning_rate,iterations,typ):
    m = len(y)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations,9))
    for it in range(iterations):
        if typ == 1:
            prediction = np.dot(X,theta)
            theta = theta -(1/m)*learning_rate*( X.T.dot((prediction - y)))
            cost_history[it]  = cal_cost_mse(theta,X,y)
            theta_history[it,:] =theta.T
        elif typ == 2:
            prediction = np.dot(X,theta)
            theta = theta -(1/m)*learning_rate*( X.T.dot((prediction - y)))
            cost_history[it]  = cal_cost_mae(theta,X,y)
            theta_history[it,:] =theta.T
        else:
            prediction = np.dot(X,theta)
            theta = theta -(1/m)*learning_rate*( X.T.dot((prediction - y)))
            cost_history[it]  = cal_cost_mape(theta,X,y)
            theta_history[it,:] =theta.T
    return theta, cost_history, theta_history

	def train(self, filename):

		data = pd.read_csv(filename, delimiter=',')

		data = data.dropna()
		summary_unique = data['Summary'].unique()
		data['Summary'] = data['Summary'].astype('category')
		data['Summary'] = data['Summary'].cat.reorder_categories(summary_unique, ordered=True)
		data['Summary'] = data['Summary'].cat.codes

		precip_unique = data['Precip Type'].unique()
		data['Precip Type'] = data['Precip Type'].astype('category')
		data['Precip Type'] = data['Precip Type'].cat.reorder_categories(precip_unique, ordered=True)
		data['Precip Type'] = data['Precip Type'].cat.codes

		daily_unique = data['Daily Summary'].unique()
		data['Daily Summary'] = data['Daily Summary'].astype('category')
		data['Daily Summary'] = data['Daily Summary'].cat.reorder_categories(daily_unique, ordered=True)
		data['Daily Summary'] = data['Daily Summary'].cat.codes


		min_max_scaler = preprocessing.MinMaxScaler()

		data = data.drop('Formatted Date', axis=1)
		x = data.loc[:, data.columns != 'Apparent Temperature (C)']
		x_data = x.values
		x_scaled = min_max_scaler.fit_transform(x_data)
		x1 = pd.DataFrame(x_scaled)

		y = data.loc[:,'Apparent Temperature (C)']
		y_data = y.values.reshape((y.shape[0],1))
		y_scaled = min_max_scaler.fit_transform(y_data)
		y1 = pd.DataFrame(y_scaled)

		lr = 0.01
		n_iter = 1000

		theta = np.random.randn(x.shape[1],1)

		self.theta1,cost_history,theta_history = self.gradient_descent(x1,y1,theta,lr,n_iter,1)
		print(self.theta1)

	def predict(self, filename):
		data = pd.read_csv(filename, delimiter=',')

		data = data.dropna()
		summary_unique = data['Summary'].unique()
		data['Summary'] = data['Summary'].astype('category')
		data['Summary'] = data['Summary'].cat.reorder_categories(summary_unique, ordered=True)
		data['Summary'] = data['Summary'].cat.codes

		precip_unique = data['Precip Type'].unique()
		data['Precip Type'] = data['Precip Type'].astype('category')
		data['Precip Type'] = data['Precip Type'].cat.reorder_categories(precip_unique, ordered=True)
		data['Precip Type'] = data['Precip Type'].cat.codes

		daily_unique = data['Daily Summary'].unique()
		data['Daily Summary'] = data['Daily Summary'].astype('category')
		data['Daily Summary'] = data['Daily Summary'].cat.reorder_categories(daily_unique, ordered=True)
		data['Daily Summary'] = data['Daily Summary'].cat.codes


		min_max_scaler = preprocessing.MinMaxScaler()

		data = data.drop('Formatted Date', axis=1)
		x = data.loc[:, data.columns != 'Apparent Temperature (C)']
		x_data = x.values
		x_scaled = min_max_scaler.fit_transform(x_data)
		x1 = pd.DataFrame(x_scaled)

		y = data.loc[:,'Apparent Temperature (C)']
		y_data = y.values.reshape((y.shape[0],1))
		y_scaled = min_max_scaler.fit_transform(y_data)
		y1 = pd.DataFrame(y_scaled)

		y_pred1 = x1.dot(self.theta1)
		
		return y_pred1