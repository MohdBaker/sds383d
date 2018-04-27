from __future__ import division 
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numdifftools as nd
import math
import pickle

df = pd.read_csv("iris.csv") #load data into pandas dataframe

#split data
y_string = df.as_matrix(columns=['Species']).ravel() #

X = df.as_matrix(columns=df.columns[0:4]) #





#map label to 1 and -1 
y_l=-1*np.ones(y_string.shape)
label1=np.where(y_string=='setosa')
label2=np.where(y_string=='versicolor')
label3=np.where(y_string=='virginica')

y_l[label1]=0
y_l[label2]=1
y_l[label3]=2


## get data fro each group
X1=X[label1,:].squeeze()
X2=X[label2,:].squeeze()
X3=X[label3,:].squeeze()
y1=y_l[label1]
y2=y_l[label2]
y3=y_l[label3]


# Remove NA


X=X.astype('float')
y_l=y_l.astype('float')
y_l=np.expand_dims(y_l, axis=1)

y=np.zeros((450,1))
y[0:50]=1
y[200:250]=1
y[400:450]=1



### add intercept
#X = np.hstack((np.ones((X.shape[0],1)),X)) #



# set zero mean 1 sigma prior
mu0=0
sigma=1

def kernel(x1, x2, sigma2, l):
    return sigma2*np.exp(-1 * (np.linalg.norm(x1-x2) ** 2) / (2*l**2))

def matrix(d,sigma2, l):
    return [[kernel(x1,x2,sigma2, l) for x2 in d] for x1 in d]


alpha2=2
l=1
K1=np.asarray(matrix(X,alpha2, l))
K= scipy.linalg.block_diag(K1,K1,K1)

K_inv=np.linalg.inv(K)

# define the objective function for the optimization 
def objective(x,y,K,K_inv):
	x=np.expand_dims(x, axis=1)
	temp=0
	for i in range(150):
		temp=temp+np.log(np.exp(x[i])+np.exp(x[i+150])+np.exp(x[i+300]))
	logis= temp  -np.dot(y.T,x)
	prior=np.dot(np.dot(x.T,0.5*K_inv),x)+0.5*np.log(np.linalg.det(K))
	return logis+prior

x0=np.zeros(y.shape)
results=minimize(objective,x0.squeeze(),args=(y,K,K_inv) , method='BFGS',options={ 'disp': True})
f=results.x
yy= np.zeros(f.shape)
for i in range(yy.shape[0]):
	ind=np.mod(i,150)	
	yy[i]=np.exp(f[i])/(np.exp(f[ind])+np.exp(f[ind+150])+np.exp(f[ind+300]))


f=np.expand_dims(f, axis=1)

f1=1/(1+np.exp(-1*f))


bins = np.linspace(-1, 1, 100)
plt.hist(f[300:350],bins, edgecolor='k',fc=(0, 0, 1, 0.5),label='group1')
plt.hist(f[350:400],bins, edgecolor='k',fc=(1, 0, 0, 0.5), label='group2')
plt.hist(f[400:450],bins, edgecolor='k',fc=(0, 1, 0, 0.5), label='group3')
plt.legend()
plt.show(block=False)


#H=nd.Hessian(objective)(f,y,K,K_inv)

#H = pickle.load( open( "H.p", "rb" ) )


