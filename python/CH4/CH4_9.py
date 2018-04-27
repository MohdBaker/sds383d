from __future__ import division 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel as W
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv("weather.csv") #load data into pandas dataframe

#split data
y = df.as_matrix(columns=['temperature']).ravel() #

X = df.as_matrix(columns=df.columns[2:4])  #


kernel = C(1.0, (1e-3, 1e3)) * RBF([0.1,0.2], (1e-2, 1e2))+W(1.0)



 

gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)


X=X.astype('float')
y=np.expand_dims(y, axis=1)
data=X
n=X.shape[0]
d=X.shape[1]

Xtrain=X[0:int(0.8*n),:]
ytrain=y[0:int(0.8*n),:]
Xtest=X[int(0.8*n)+1:,:]
ytest=y[int(0.8*n)+1:,:]

gp.fit(Xtrain, ytrain)
lik, grad=gp.log_marginal_likelihood(theta=[1.0,1.0,1.0,1.0],  eval_gradient=True)

def obj(theta):
	a,b=gp.log_marginal_likelihood(theta, True)
	return -1*a,-1*b

n_iter=1
theta=np.zeros((4,n_iter))
x0=[0,0,0,0]
theta[:,0], f, c=scipy.optimize.fmin_tnc(obj, x0)

	
#for i in range(n_iter):
#	theta[:,i], f, c=scipy.optimize.fmin_tnc(obj, x0)
#	x0=theta[:,i]

## prediction
kernel1 = C(theta[0,0], (1e-3, 1e3)) * RBF(theta[1:3,0], (1e-2, 1e2))+W(theta[3,0])
gp2 = GaussianProcessRegressor(kernel=kernel1, n_restarts_optimizer=9)

y_pred, sigma = gp2.predict(Xtest, return_std=True)



# plots

#plt.figure()
#bins = np.linspace(-2, 5, 100)

#plt.hist(theta[0,:],bins, edgecolor='k',label='alpha2')
#plt.hist(theta[1,:],bins,edgecolor='k', label='l1')
#plt.hist(theta[2,:],bins,edgecolor='k', label='l2')
#plt.hist(theta[3,:], bins,edgecolor='k', label='delta')
#plt.legend(loc='upper right')
#plt.show(block=False)



l1=np.min(X[:,0])
l2=np.min(X[:,1])
h1=np.max(X[:,0])
h2=np.max(X[:,1])


xx=np.zeros((200,2))
xx[:,0]=np.linspace(l1,h1, 200)
xx[:,1]=np.linspace(l2,h2, 200)
yy, sigma = gp.predict(xx, return_std=True)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Xtrain[:,0].squeeze(),ytrain.squeeze(),Xtrain[:,1].squeeze(),'ro', label='train')
ax.scatter(Xtest[:,0].squeeze(), ytest.squeeze(),Xtest[:,1].squeeze(),'bs', label='test')
ax.legend(['train','test'])
ax.set_xlabel('lon')
ax.set_ylabel('temp')
ax.set_zlabel('lat')

plt.show(block=False)



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Xtrain[:,0].squeeze(),ytrain.squeeze(),Xtrain[:,1].squeeze(),'ro', label='train')
ax.scatter(Xtest[:,0].squeeze(), ytest.squeeze(),Xtest[:,1].squeeze(),'bs', label='test')
ax.scatter(xx[:,0].squeeze(), yy.squeeze(),xx[:,1].squeeze(),'b-', label='mean')
ax.legend(['train','test', 'mean'])
ax.set_xlabel('lon')
ax.set_ylabel('temp')
ax.set_zlabel('lat')

plt.show()





x1 = np.linspace(Xtrain[:,0].min(),Xtrain[:,0].max(),100)
x2 = np.linspace(Xtrain[:,1].min(),Xtrain[:,1].max(),100)


Z=np.zeros((100,100))
for i in range (100):
	for j in range (100):
		x=np.asarray([x1[i],x2[j]])
		x=np.expand_dims(x, axis=1)
		Z[99-i,99-j] = gp.predict(x.T, return_std=False)


fig=plt.figure()
plt.imshow(Z.T, cmap='hot', interpolation='nearest')
plt.show()






