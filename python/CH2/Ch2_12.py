from __future__ import division 
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

df = pd.read_csv("dental.csv") #load data into pandas dataframe

#split data
y = df.as_matrix(columns=['distance']).ravel() 	# get the y values

X = df.as_matrix(columns=df.columns[2:5]) 	# get the x values

X = np.hstack((np.ones((X.shape[0],1)),X)) 	# add an intercept 


## Change the Male/Female label to 0/1
male=np.where(X[:,-1]=="Male")				
female=np.where(X[:,-1]=="Female")
X[male,-1]=0
X[female,-1]=-1


##### Remove the subject covariate 
X=np.delete(X,2,1)

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
Y1=np.dot(X,betahat_LS)
Y2=np.dot(X,betahat_BM)
Y3=np.dot(X,betahat_R)
plt.plot(y,'ro')
plt.plot(Y1,'g^')
plt.plot(Y2,'bs')
plt.plot(Y3,'y*')
plt.legend(['golden', 'LS', 'BM','R'])
plt.show()


#Now compare to least squares

results = sm.OLS(y,X).fit()
print(results.summary()) 

