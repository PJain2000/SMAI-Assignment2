import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import pickle
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
np.random.seed(0)

def load(name):
    file = open(name,'rb')
    data = pickle.load(file)
    file.close()
    return data

def save(data,name):
    file = open(name, 'wb')
    pickle.dump(data,file)
    file.close()

class GMM1D:
    def __init__(self,X,iterations,initmean,initprob,initvariance):
    # initmean = [a,b,c] initprob=[1/3,1/3,1/3] initvariance=[d,e,f]
        self.iterations = iterations
        self.X = X
        self.mu = initmean
        self.pi = initprob
        self.var = initvariance

    """E step"""

    def calculate_prob(self,r):
        for c,g,p in zip(range(3),[norm(loc=self.mu[0],scale=self.var[0]),
                                   norm(loc=self.mu[1],scale=self.var[1]),
                                   norm(loc=self.mu[2],scale=self.var[2])],self.pi):
            r[:,c] = p*g.pdf(self.X)
        """
        Normalize the probabilities such that each row of r sums to 1 and weight it by mu_c == the fraction of points belonging to 
        cluster c
        """
        for i in range(len(r)):
            # Write code here
            a = np.sum(self.pi)
            b = np.sum(r,axis=1)
            c = a*b
            r[i] = r[i]/c[i]
            pass
        return r

    def plot(self,r):
        fig = plt.figure(figsize=(10,10))
        ax0 = fig.add_subplot(111)
        for i in range(len(r)):
            ax0.scatter(self.X[i],0,c=np.array([r[i][0],r[i][1],r[i][2]]),s=100)
        """Plot the gaussians"""
        for g,c in zip([norm(loc=self.mu[0],scale=self.var[0]).pdf(np.linspace(-20,20,num=60)),
                        norm(loc=self.mu[1],scale=self.var[1]).pdf(np.linspace(-20,20,num=60)),
                        norm(loc=self.mu[2],scale=self.var[2]).pdf(np.linspace(-20,20,num=60))],['r','g','b']):
            ax0.plot(np.linspace(-20,20,num=60),g,c=c)

    def run(self):
        
        for iter in range(self.iterations):

            """Create the array r with dimensionality nxK"""
            r = np.zeros((len(self.X),3))  

            """
            Probability for each datapoint x_i to belong to gaussian g 
            """
            r = self.calculate_prob(r)


            """Plot the data"""
            self.plot(r)
            
            """M-Step"""

            """calculate m_c"""
            m_c = []
            # write code here
            for i in range(r.shape[1]):
                m_c.append(np.sum(r[:,i]))
            
            """calculate pi_c"""
            # write code here
            for i in range(len(m_c)):
                l = np.sum(m_c)
                self.pi[i] = m_c[i]/l
            
            """calculate mu_c"""
            # write code here
            self.reshape_X  = self.X.reshape(len(self.X),1)
            self.mu = np.sum(self.reshape_X*r, axis=0)/m_c


            """calculate var_c"""
            var_c = []
            #write code here
            for i in range(r.shape[1]):
                self.reshape_r = np.array(r[:,i]).reshape(180,1)
                a = (self.reshape_r*self.reshape_X-self.mu[i])
                b = self.reshape_X-self.mu[i]
                obj_append = (1/m_c[i]) * np.dot(a.T,b)
                var_c.append(obj_append)
            plt.show()

data1 = load("./Datasets/Question-2/dataset1.pkl")
data2 = load("./Datasets/Question-2/dataset2.pkl")
data3 = load("./Datasets/Question-2/dataset3.pkl")

data = np.stack((data1,data2,data3)).flatten()

mean1 = np.mean(data1)
mean2 = np.mean(data2)
mean3 = np.mean(data3)

var1 = 2
var2 = 1.2
var3 = 1

g = GMM1D(data,10,[mean1,mean2,mean3],[1/3,1/3,1/3],[var1,var2,var3])

g.run()

   
data_re = data.reshape(data.shape[0],1)
gmm = GaussianMixture(n_components=3).fit(data_re)
labels = gmm.predict(data_re)
# print(labels)

plt.scatter(data_re[:, 0], np.zeros_like(data_re[:, 0]), c=labels, s=40, cmap='viridis');
plt.show()