from __future__ import division 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("restaurants.csv") #load data into pandas dataframe

#split data
y = df.as_matrix(columns=['Profit']).ravel() 	# get the y values

X = df.as_matrix(df.columns[3:])	# get the x values

label=df.as_matrix(columns=['DinnerService']).ravel()

X = np.hstack((np.ones((X.shape[0],1)),X)) 	# add an intercept 

ymax=np.max(y)
ymin=np.min(y)
y=np.divide((y-ymin),(ymax-ymin))




## Chanage Matrix X to float
X=X.astype('float')

#set the params
n=X.shape[0]		# number of samples 
d=X.shape[1]		# number of covariates 



# change y to (N,1) instead of (N,)
y=np.expand_dims(y, axis=1)

#



mu0=0.5
k0=3
a0=1
b0=1

def sample_mean_sigma(y,k0,mu0,n,a0,b0):
	if n==0:
		mu_n=mu0
		k_n=k0
		a=a0
		b=b0
	else:	
		mu_n=(k0*mu0+n*np.mean(y))/(k0+n)
		k_n=k0+n
		a=a0+n/2
		b=b0+0.5*np.sum(y**2)+0.5*(k0*n*np.sum((y-mu0)**2)/((2*n)*(k0+n)))
	prec=np.random.gamma(a,1/b)
	sigm=1/(prec*k_n)
	mu=np.random.normal(mu_n,sigm )
	return mu, sigm


def sample_z(y,m1,m0,sigma1,sigma0, n):
	z=np.zeros((n,1))
	for i in range (n):
		p1=np.exp(-1*(y[i]-m1)**2/(2*sigma1))
		p0=np.exp(-1*(y[i]-m0)**2/(2*sigma0))
		if p1>p0:
			z[i]=1
		else:
			z[i]=0
	return z.squeeze() 

ind1=np.where(y>0.5)
z0=np.zeros(y.shape)
z0[ind1]=1
def sampler(y,mu0,a0,k0,b0,z0,n,n_samples=1000):
	m0=np.zeros((n_samples,1))
	m1=np.zeros((n_samples,1))
	sigm0=np.zeros((n_samples,1))
	sigm1=np.zeros((n_samples,1))
	z=np.zeros((n, n_samples))
	z[:,0]=z0.squeeze()
	for i in range(1,n_samples):
		ind1=np.where(z[:,i-1]==1)
		ind0=np.where(z[:,i-1]==0)
		y1=y[ind1]	
		y0=y[ind0]
		#print(i)
		m0[i],sigm0[i]=sample_mean_sigma(y0,k0,mu0,y0.shape[0],a0,b0)
		m1[i],sigm1[i]=sample_mean_sigma(y1,k0,mu0,y1.shape[0],a0,b0)
		#print('m0 %.6f, sigma0 %.6f    m1 %.6f, sigma1 %.6f' %(m0[i], sigm0[i],m1[i],sigm1[i]))
		z[:,i]=sample_z(y,m1[i],m0[i],sigm1[i],sigm0[i], n)
	return m0,m1,z


m0,m1,z=sampler(y,mu0,a0,k0,b0,z0,n,1500)
m0_final=np.mean(m0[500:])
m1_final=np.mean(m1[500:])
z_final=np.mean(z[:,500:],axis=1)
z_label=np.around(z_final)
n_mismatch=np.nonzero(np.abs(label-z_label))[0].shape[0]


