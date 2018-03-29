from __future__ import division 
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scipy.stats as stats

df = pd.read_csv("pima.csv") #load data into pandas dataframe

#split data
y_l = df.as_matrix(columns=['class_variable']).ravel() 	#  get labels 

X = df.as_matrix(columns=df.columns[0:10]) 		#  get covariates

X = np.hstack((np.ones((X.shape[0],1)),X)) 		#  add intercept 


y_l=np.expand_dims(y_l, axis=1)
data=X

## set params
n=X.shape[0]   	# number of samples
d=X.shape[1]	# dimensions
K=np.eye(d)	# intialize to identity 
A=np.eye(n)	# intialize to identity 
mu=np.ones((d,1))	# intialize to ones 
a=0.5
b=0.5
alpha=0.5
mu0=0.5*np.ones((d,1))
tau=0.5


########################################### NOTE: 'y' CORRESPONDS TO 'z' IN THIS CODE WHILE 'y_l' CORRESPONDS TO THE LABEL ######################
 
# define sampler function 
def sampler(X,y_l,n,a,b,A, K,mu0,num_samples=1000):
	# initalize the variables to samples (space allocation)
	omega = np.ones(num_samples)
	beta=np.random.randn(d,num_samples)
        y=np.zeros(y_l.shape)
	# intialize the values for K_n, a_n, and b_n for the first iteration 
        k_n=K
        a_n=a
        b_n=b
        mu_n=mu0
        y=y_l-0.75  # initialize z to some values consistent with the right label, postive values for y_l=1, negative for y_l=0 

        for i in range(num_samples):
            # print iteration number
            print ('############################## ',i,'   ############################################')
            # update params
	    k_n = np.dot(np.dot(X.T,A),X)+K
	    mu_n = np.dot(np.linalg.inv(k_n),(np.dot(K,mu0)+np.dot(np.dot(X.T,A),y)))
	    a_n = a + n/2
	    b_n = b+0.5*(-1*np.dot(np.dot(mu_n.T,k_n),mu_n)+np.dot(np.dot(mu0.T,K),mu0)+np.dot(np.dot(y.T,A),y))
	    # sample omega
	    omega[i] =np.random.gamma(shape=a_n, scale=1/b_n)
	    # sample beta
	    beta[:,i] = np.random.multivariate_normal(np.squeeze(mu_n),np.linalg.inv(omega[i]*k_n), 1)
            # sample z
	    y=sample_y(y_l,beta[:,i],X,omega[i])
		

    	return y, omega,beta

def sample_y(y_l,beta,X,omega ):
    y=np.zeros(y_l.shape)
    for i in range(y_l.shape[0]):
	me= np.dot(X[i,:],beta)
	cv=1/omega
	std=np.sqrt(cv)
        if y_l[i]==1:
        	trun_a=0
		dist=stats.truncnorm( (trun_a - me) / std,100 , loc=me, scale=std)
		y[i]=dist.rvs(1)
	elif y_l[i]==0:
        	trun_b=0
		dist=stats.truncnorm( -100,(trun_b - me) / std , loc=me, scale=std)        	
		y[i]=dist.rvs(1)
    return y
# normalize X, this will help in choosing the starting point
X_normed = X / X.max(axis=0)

burn=500
n_iter=2*burn    
# run sampler        
[y,omega, beta]=sampler(X_normed,y_l,n,a,b,A, K,mu0,n_iter)
BETA=np.mean(beta[:,burn:],1)
BETA=np.expand_dims(BETA, axis=1)

## get the labels from the latent var
y_hat=np.sign(np.dot(X_normed,BETA))


# set zero labels for negative latent predictions
neg=np.where(y_hat==-1)
y_hat[neg]=0

# compute number of errors 
errors=np.count_nonzero(y_l-y_hat)

# accuracy value
acc=(n-errors)/n



