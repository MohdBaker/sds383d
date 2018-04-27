from __future__ import division 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import wishart,multivariate_normal

df = pd.read_csv("mnist.csv",header=None) #load data into pandas dataframe

#split data

X = df.as_matrix()	
X=X.astype('float')
pca = PCA(n_components=50)
pca.fit(X)
X=pca.transform(X)

#set the params
n=X.shape[0]		# number of samples 
d=X.shape[1]		# number of covariates 
K=10

#



mu0=np.ones((d,1))
k0=3
nu0=1
L0=np.eye(d)


alpha=np.ones((K,1))


def sample_mean_sigma(X,k0,nu0,L0,n):
	if n==0:
		mu_n=mu0
		k_n=k0
		nu=nu0
		L=L0
	else:	
		mu_n=(k0*mu0.squeeze()+n*np.mean(X,axis=0))/(k0+n)
		k_n=k0+n
		nu=nu0+n
		L=L0+0.5*(np.dot(X.T,X)-k_n*np.dot(mu_n,mu_n.T))
	sigm=wishart.rvs(df=nu,scale=(np.linalg.inv(L)+0.0001*np.eye(mu0.shape[0])),size=1,random_state=None)
	mu=np.random.multivariate_normal(mu_n,np.divide(sigm,k_n),1 ).T
	return mu.squeeze(), sigm


def sample_z(X,mu,sigma,alpha,K, n):
	z=np.zeros((n,1))
	p=np.zeros((n,K))
	alpha_n=np.zeros((K,))
	for i in range (n):
		for k in range(K):
			prob=multivariate_normal(mean=mu[:,k].squeeze(),cov=sigma[:,:,k].squeeze()+0.0001*np.eye(mu.shape[0]))
			p[i,k]=prob.pdf(X[i,:])			
			alpha_n[k]=alpha[k]+np.where(z==k)[0].shape[0]
		z[i]=np.argmax(np.multiply(p[i,:],alpha_n))
	return z.squeeze() 


z0=np.random.randint(0,high=10,size=n)
def sampler(y,mu0,k0,nu0,L0,alpha,n,K,n_samples=1000):
	m=np.zeros((d,K,n_samples))
	sigm=np.zeros((d,d,K,n_samples))
	z=np.zeros((n, n_samples))
	z[:,0]=z0.squeeze()
	for i in range(1,n_samples):
		for k in range(K):
			ind=np.where(z[:,i-1]==k)
			x=X[ind[0],:]
			print('number is %.6f' %(ind[0].shape) )
			m[:,k,i],sigm[:,:,k,i]=sample_mean_sigma(x,k0,nu0,L0,x.shape[0])
		z[:,i]=sample_z(X,m[:,:,i],sigm[:,:,:,i],alpha,K, n)
		print('sum of z %.6f'% (np.sum(z[:,i])))
	return m,z


m,z=sampler(X,mu0,k0,nu0,L0,alpha,n,K,n_samples=3)
z_final=np.mean(z[:,500:],axis=1)
z_label=np.around(z_final)
n_mismatch=np.nonzero(np.abs(label-z_label))[0].shape[0]


