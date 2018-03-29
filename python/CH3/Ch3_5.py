from __future__ import division 
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt


df = pd.read_csv("titanic.csv") #load data into pandas dataframe

#split data
y_string = df.as_matrix(columns=['Survived']).ravel() #

X = df.as_matrix(columns=['Age']).ravel() #
#map label to 1 and -1 
y_l=-1*np.ones(y_string.shape)
label1=np.where(y_string=='Yes')
y_l[label1]=1

# Remove NA
valid=np.where(X>0)
y_l=y_l[valid]
X=X[valid]
X=X.astype('float')
y_l=y_l.astype('float')
y_l=np.expand_dims(y_l, axis=1)
X=np.expand_dims(X, axis=1)

### add intercept
X = np.hstack((np.ones((X.shape[0],1)),X)) #



# set zero mean 1 sigma prior
mu0=np.zeros((2,1))
sigma=1

#define the objective fucntion to minimize
def objective(x,X,y_l,mu0,sigma):
	x=np.expand_dims(x, axis=1)
	logis=np.sum(np.log(1+np.exp(-y_l*np.dot(X,x))))
	prior=0.5*np.dot((x-mu0).T,(x-mu0))/sigma**2
	return logis+prior
# set intial guess
x0=np.ones((2,))
# get MAP estimate
results=minimize(objective,x0,args=(X,y_l,mu0,sigma) , method='nelder-mead',options={'xtol': 1e-8, 'disp': True})
beta=results.x
print('############## BETA:', beta, '################')




beta=np.expand_dims(beta, axis=1)
## evaluate the posterior of beta
def eval_post(beta, X,y_l,mu0,sigma):
	lik=np.prod(1/(1+np.exp(-y_l*np.dot(X,beta))))
	prior=np.exp(-(0.5/sigma)*np.dot((beta-mu0).T,(beta-mu0)))
	return lik*prior
P_beta=eval_post(beta, X,y_l,mu0,sigma)


## compute hessian 
H11= np.sum(y_l**2*X[:,0]**2*np.exp(y_l*np.dot(X,beta))/(1+np.exp(y_l*np.dot(X,beta)))**2)+1/sigma
H22= np.sum(y_l**2*X[:,1]**2*np.exp(y_l*np.dot(X,beta))/(1+np.exp(y_l*np.dot(X,beta)))**2)+1/sigma
H21= np.sum(y_l**2*X[:,0]*X[:,1]*np.exp(y_l*np.dot(X,beta))/(1+np.exp(y_l*np.dot(X,beta)))**2)






