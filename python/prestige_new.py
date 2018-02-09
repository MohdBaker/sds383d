from __future__ import division 
import numpy as np
import pandas as pd
import statsmodels.api as sm


df = pd.read_csv("prestige.csv") #load data into pandas dataframe

#split data
y = df.as_matrix(columns=['prestige']).ravel() #column 'prestige', as a 1d numpy array

X = df.as_matrix(columns=df.columns[1:4]) #education, income and percent women, as a 2d numpy array

X = np.hstack((np.ones((X.shape[0],1)),X)) #add an intercept

#compute the estimator

betahat = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),y)

# Fill in the blanks
y= np.expand_dims(y, axis=1)              # change to Nx1 matrix
betahat= np.expand_dims(betahat, axis=1)  # change to Nx1 matrix
yhat=np.dot(X,betahat)                    # get prediction for y
sigma2=np.sum((yhat-y)**2)/(X.shape[0]-X.shape[1])  # get the unbaised estimator for sigma2 
betacov = np.linalg.inv(np.dot(X.T,X))*sigma2 # based on the equation obtained from exercise 1.23

se_beta = np.sqrt(np.diag(betacov))
#Now compare to least squares

results = sm.OLS(y,X).fit()
print(results.summary()) 
print('\n\n********Computed Standard Error Values**************\n')
print(se_beta)
print('\n********Computed Standard Error Values**************\n')
