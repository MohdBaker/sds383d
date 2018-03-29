from __future__ import division 
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import math

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
#X = np.hstack((np.ones((X.shape[0],1)),X)) #



# set zero mean 1 sigma prior
mu0=0
sigma=1

# define the objective function for the optimization 
def objective(x,X,y_l,mu0,sigma):
	logis=np.sum(np.log(1+np.exp(-y_l*x*X)))
	prior=0.5*(x-mu0)**2/sigma**2
	return logis+prior

x0=1
results=minimize(objective,x0,args=(X,y_l,mu0,sigma) , method='nelder-mead',options={'xtol': 1e-8, 'disp': True})
beta=results.x
print('############## BETA:', beta[0], '################')



## evaluate the posterior of beta
def eval_post(beta, X,y_l,mu0,sigma):
	lik=np.prod(1/(1+np.exp(-y_l*beta*X)))
	prior=np.exp(-(0.5/sigma)*(beta-mu0)**2)
	return lik*prior

# generate a range of betas around the MAP solution to evaluate the posterior 
betas=np.linspace(beta-0.05,beta+0.05,100)

# create vector of same size 
prob=np.zeros(betas.shape)

#evaluate the post at the range of betas
for i in range(betas.shape[0]):
	prob[i]=eval_post(betas[i], X,y_l,mu0,sigma)
fig, ax1 = plt.subplots()
ax2=plt.twinx()

# plot samples from the posterior 
ax1.plot(betas,prob,'r*')
ax1.set_xlabel('Beta Values')
ax1.set_ylabel('Unnorm Prob')

# laplace approximation 
#compute the value of c
C=np.sum(y_l**2*X**2*np.exp(y_l*beta*X)/(1+np.exp(y_l*beta*X))**2)+1/sigma
#get sigma form c
sigma_new=np.sqrt(1/C)

# plot the posterior approximation 
ax2.plot(betas,mlab.normpdf(betas, beta, sigma_new))
ax2.set_ylabel('Unnorm Prob')
ax1.legend(['Posterior'],loc="upper left")
ax2.legend(['Approx'],loc="upper right")
plt.show()



plt.show()



