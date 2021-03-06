from __future__ import division 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import wishart,multivariate_normal
from sklearn.preprocessing import scale

df = pd.read_csv("mnist.csv",header=None) #load data into pandas dataframe

#split data

Xp = df.as_matrix()	
Xp=Xp.astype('float')
Xp=scale( Xp, axis=0, with_mean=True, with_std=True, copy=True )
pca = PCA(n_components=50)
pca.fit(Xp)
X=pca.transform(Xp)

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


def sample_mean(X,mu0,n):
	d=X.shape[1]
	sigma=np.linalg.inv((n+1)*np.eye(d))	
	mu_n=np.dot(sigma, ((np.dot(np.eye(d),mu0.squeeze()))+n*np.dot(np.eye(d),np.mean(X,axis=0)) ))
	mu=np.random.multivariate_normal(mu_n,sigma).T
	return mu.squeeze()


def sample_z(X,mu,alpha,K, n):
	z=np.zeros((n,1))
	p=np.zeros((n,K))
	alpha_n=np.zeros((K,))
	for i in range (n):
		for k in range(K):
			prob=multivariate_normal(mean=mu[:,k].squeeze(),cov=np.eye(mu.shape[0]))
			p[i,k]=prob.pdf(X[i,:])			
			alpha_n[k]=alpha[k]+np.where(z==k)[0].shape[0]
		P=np.divide(np.multiply(p[i,:],alpha_n), np.sum(np.multiply(p[i,:],alpha_n)))
		#z[i]=np.argmax(np.multiply(p[i,:],alpha_n))  #sample from multi here		
		z[i]=np.argmax(np.random.multinomial(1,P))
	return z.squeeze() 


z0=np.random.randint(0,high=10,size=n)
#for i in range(n):
#	z0[i]=int(i/100)

def sampler(X,mu0,z0,alpha,n,K,n_samples=1000):
	m=np.zeros((d,K,n_samples))
	z=np.zeros((n, n_samples))
	z[:,0]=z0.squeeze()
	for i in range(1,n_samples):
		for k in range(K):
			ind=np.where(z[:,i-1]==k)
			x=X[ind[0],:]
			print('number is %.6f' %(ind[0].shape) )
			m[:,k,i]=sample_mean(x,mu0,x.shape[0])
		z[:,i]=sample_z(X,m[:,:,i],alpha,K, n)
		print('Iteration %.6f '% (i))
	return m,z


m,z=sampler(X,mu0,z0,alpha,n,K,n_samples=3000)
MM=np.mean(m[:,:,2000:], axis=2)
M=pca.inverse_transform(MM.T)

fig, ax =plt.subplots(nrows=2, ncols=5)
c=0
for i in range(2):
	for j in range(5):
		ax[i, j].imshow(M[c,:].reshape(28,28))
		c+=1

plt.show()




### final results:
#number is 128.000000
#number is 75.000000
#number is 89.000000
#number is 40.000000
#number is 31.000000
#number is 147.000000
#number is 75.000000
#number is 85.000000
#number is 110.000000
#number is 220.000000



