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
X[male,-1]=1
X[female,-1]=0

##### Remove the subject covariate 
X=np.delete(X,2,1)

## Chanage Matrix X to float
X=X.astype('float')

# change y to (N,1) instead of (N,)
y=np.expand_dims(y, axis=1)
data=X

#set the params
N=X.shape[0]		# number of samples 
d=X.shape[1]		# number of covariates 
K=np.eye(d)             # matrxi K as identity 
A=np.eye(N)		# matrix A (coressponding to Lambda) as identity 
mu=np.ones((d,1))	# meam vector of 1 
a=1			# gamma distribution params
b=1

alpha=0.5
mu0=0.5*np.ones((d,1)) 	# initialize mu0
tau=0.5

# define a samples function 
def sampler(X,y,n,a,b,tau,A, K,mu0,num_samples=1000):
        # intialize empty matrices for metric to sample
	lambd = np.zeros((num_samples,n)) 
	omega = np.zeros(num_samples)
	beta=np.zeros((d,num_samples))
	for i in range(num_samples):
		# compute updated params
		k_n = np.dot(np.dot(X.T,A),X)+K
		mu_n = np.dot(np.linalg.inv(k_n),(np.dot(K,mu0)+np.dot(np.dot(X.T,A),y)))
		a_n = a + n/2
		b_n = b+0.5*(-1*np.dot(np.dot(mu_n.T,k_n),mu_n)+np.dot(np.dot(mu0.T,K),mu0)+np.dot(np.dot(y.T,A),y))
		# sample omega		
		omega[i] =np.random.gamma(shape=a_n, scale=1/b_n)
		# sample beta
		beta[:,i] = np.random.multivariate_normal(np.squeeze(mu_n),np.linalg.inv(omega[i]*k_n), 1)
		# sample lambda
		for j in range(n):
			lambd[i,j]=np.random.gamma(shape=tau+0.5,scale=1/(0.5*omega[i]*(y[j]-np.dot(X[j,:],beta[:,i]))**2+tau))
		A=np.diag(lambd[i,:])

    	return lambd, omega,beta
## estimator for male-only model
[lambd_M, omega_M,beta_M]=sampler(X[0:64,:],y[0:64],64,a,b,tau,A[0:64,0:64], K,mu0,5000)

## estimator for female-only model
[lambd_F, omega_F,beta_F]=sampler(X[64:,:],y[64:],44,a,b,tau,A[64:,64:], K,mu0,5000)

## estimator for single model  
[lambd, omega,beta]=sampler(X,y,n,a,b,tau,A, K,mu0,5000)

## assume 1000 burn in runs, and avergae the other samples 
betahat_HM=np.mean(beta[:,1000:-1],1)
betahat_HM_M=np.mean(beta_M[:,1000:-1],1)
betahat_HM_F=np.mean(beta_F[:,1000:-1],1)

# compare to other methods (least square, ridge, bayesian)
betahat_LS = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),y)
betahat_R = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)+0.5*np.eye(d)),X.T),y)
term1=np.linalg.inv(np.dot(np.dot(X.T,A),X)+K)
term2=np.dot(np.dot(X.T,A),y)+np.dot(K,mu)
betahat_BM = np.dot(term1,term2)

## plots 
print'betahat_LS:',betahat_LS
print'betahat_R:',betahat_R
print'betahat_BM:',betahat_BM
print'betahat_HM:',betahat_HM
Y1=np.dot(X,betahat_LS)
Y2=np.dot(X,betahat_BM)
Y3=np.dot(X,betahat_R)
Y4=np.dot(X,betahat_HM)
Y5=np.dot(X[0:64,:],betahat_HM_M)
Y6=np.dot(X[64:,:],betahat_HM_F)
plt.plot(y,'ro')
plt.plot(Y1,'g^')
plt.plot(Y2,'bs')
plt.plot(Y3,'y*')
plt.plot(Y4,'ys')
plt.plot(Y5,'rs')
plt.plot(np.arange(64,108),Y6,'gs')
plt.legend(['golden', 'LS', 'BM','R','HM','HM_Male','HM_Female'])
#plt.legend(['golden', 'HM','HM_Male', 'HM_Female'])
plt.show()






