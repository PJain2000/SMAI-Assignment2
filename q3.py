import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing


class Airfoil:
    theta = []
    def  cal_cost(self, theta, X, y):
        m = len(y)
        
        predictions = X.dot(theta)
        cost = (1/2*m) * np.sum(np.square(predictions-y))
        return cost

    def gradient_descent(self, X,y,theta1,learning_rate=0.01,iterations=1000):
        m = len(y)
        cost_history = np.zeros(iterations)
        theta_history = np.zeros((iterations,5))
        for it in range(iterations):
            prediction = np.dot(X,theta1)
            theta1 = theta1 -(1/m)*learning_rate*( X.T.dot((prediction - y)))
            theta_history[it,:] =theta1.T
            cost_history[it]  = self.cal_cost(theta1,X,y)
            
        return theta1, cost_history, theta_history

    def train(self, filename):

        data = pd.read_csv(filename, delimiter=',')

        min_max_scaler = preprocessing.MinMaxScaler()

        x = data.iloc[:,:-1]
        x_data = x.values
        x_scaled = min_max_scaler.fit_transform(x_data)
        x1 = pd.DataFrame(x_scaled)

        y = data.iloc[:,-1]
        y_data = y.values.reshape((y.shape[0],1))
        x_scaled = min_max_scaler.fit_transform(y_data)
        y1 = pd.DataFrame(x_scaled)

        lr = 0.01
        n_iter = 1000

        theta1 = np.random.randn(x.shape[1],1)

        # X_b = np.c_[np.ones((len(X),1)),X]
        self.theta,cost_history,theta_history = self.gradient_descent(x1,y1,theta1,lr,n_iter)
        print(self.theta)

    def predict(self, filename):
        data = pd.read_csv(filename, delimiter=',')

        min_max_scaler = preprocessing.MinMaxScaler()

        x = data.iloc[:,:-1]
        x_data = x.values
        x_scaled = min_max_scaler.fit_transform(x_data)
        x1 = pd.DataFrame(x_scaled)

        y_predicted = x1.dot(self.theta)

        return y_predicted





