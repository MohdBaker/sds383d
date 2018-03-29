from __future__ import division 
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import math

df = pd.read_csv("tea_discipline_oss.csv") #load data into pandas dataframe

#split data
y_l = df.as_matrix(columns=['ACTIONS']).ravel() #

X = df.as_matrix(columns=['GRADE']).ravel() #



# Remove -99
valid=np.where(y_l>0)
y_l=y_l[valid]
X=X[valid]
X=X.astype('float')
#X=X**2  # get GRADE^2
y_l=y_l.astype('float')
y_l=np.expand_dims(y_l, axis=1)
X=np.expand_dims(X, axis=1)

### add intercept
X = np.hstack((np.ones((X.shape[0],1)),X)) #



# set zero mean 1 sigma prior
mu0=np.zeros((2,1))
sigma=1

def objective(x,X,y_l,mu0,sigma):
	x=np.expand_dims(x, axis=1)
	pois=np.sum(-1*y_l*np.dot(X,x)+(np.exp(np.dot(X,x))))
	prior=0.5*np.dot((x-mu0).T,(x-mu0))/sigma**2
	return pois+prior

x0=np.ones((2,))
results=minimize(objective,x0,args=(X,y_l,mu0,sigma) , method='nelder-mead',options={'xtol': 1e-8, 'disp': True})
beta=results.x
print('############## BETA:', beta, '################')




beta=np.expand_dims(beta, axis=1)
## evaluate the posterior of beta
def eval_post(beta, X,y_l,mu0,sigma):
	lamb=np.exp(np.dot(X,beta))
	lik=np.prod(lamb**y_l * np.exp(-lamb) )
#	fact=1	
#	for i in range(y_l.shape[0]):
#		fact=fact*math.factorial(y_l[i])
#	lik=lik/fact
	prior=np.exp(-(0.5/sigma)*np.dot((beta-mu0).T,(beta-mu0)))
	return lik*prior
#P_beta=eval_post(beta, X,y_l,mu0,sigma)


## compute hessian 
H11= np.sum(X[:,0]**2*np.exp(np.dot(X,beta)))+1/sigma
H22= np.sum(X[:,1]**2*np.exp(np.dot(X,beta)))+1/sigma
H21= np.sum(X[:,1]*X[:,0]*np.exp(np.dot(X,beta)))






