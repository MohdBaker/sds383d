from __future__ import division 
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

df = pd.read_csv("restaurants.csv") #load data into pandas dataframe

#split data
y = df.as_matrix(columns=['Profit']).ravel() 	# get the y values

X = df.as_matrix(df.columns[2:])	# get the x values

X = np.hstack((np.ones((X.shape[0],1)),X)) 	# add an intercept 

ymax=np.max(y)
ymin=np.min(y)
y=np.divide((y-ymin),(ymax-ymin))




## Chanage Matrix X to float
X=X.astype('float')

#set the params
N=X.shape[0]		# number of samples 
d=X.shape[1]		# number of covariates 
K=np.eye(d)             # matrxi K as identity 
A=np.eye(N)		# matrix A (coressponding to Lambda) as identity 
mu=np.ones((d,1))	# meam vector of 1 
a=1			# gamma distribution params
b=1


# change y to (N,1) instead of (N,)
y=np.expand_dims(y, axis=1)

# least sqaure estimator 
betahat_LS = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),y)

# Ridge estimator 
betahat_R = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)+0.5*np.eye(d)),X.T),y)

# Bayesian Model estimator 
term1=np.linalg.inv(np.dot(np.dot(X.T,A),X)+K)
term2=np.dot(np.dot(X.T,A),y)+np.dot(K,mu)
betahat_BM = np.dot(term1,term2)

# print estimator values 
print'betahat_LS:',betahat_LS
print'betahat_R:',betahat_R
print'betahat_BM:',betahat_BM

## plot all theww models

res=y-np.dot(X,betahat_BM)
bins = np.linspace(-1, 1, 50)
plt.hist(res,bins, edgecolor='k',fc=(0, 0, 1, 0.5),label='res')
plt.hist(y,bins, edgecolor='k',fc=(1, 0, 0, 0.5),label='profit')
plt.legend()
plt.show(block=False)


