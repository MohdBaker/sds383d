from __future__ import division 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel as W

df = pd.read_csv("faithful.csv") #load data into pandas dataframe

#split data
y = df.as_matrix(columns=['eruptions']).ravel() #

X = df.as_matrix(columns=['waiting']).ravel() #
X=np.expand_dims(X, axis=1)

kernel = C(1.0, (1e-3, 1e3)) * RBF(0.1, (1e-2, 1e2))+W(1.0)



 

gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)


X=X.astype('float')
y=np.expand_dims(y, axis=1)
data=X
n=X.shape[0]
d=X.shape[1]

Xtrain=X[0:int(0.75*n),:]
ytrain=y[0:int(0.75*n),:]
Xtest=X[int(0.75*n)+1:,:]
ytest=y[int(0.75*n)+1:,:]

gp.fit(Xtrain, ytrain)
lik, grad=gp.log_marginal_likelihood(theta=[1.0,1.0,1.0],  eval_gradient=True)

def obj(theta):
	a,b=gp.log_marginal_likelihood(theta, True)
	return -1*a,-1*b

theta, f, c=scipy.optimize.fmin_tnc(obj, x0=[0.1,0.1,0.1])


alpha2=theta[0]

l=theta[1]

delta=theta[2]

## prediction

y_pred, sigma = gp.predict(Xtest, return_std=True)



# plots
plt.figure()
plt_vals = []

plt_vals.extend([Xtrain, ytrain, 'ro'])
plt_vals.extend([Xtest, ytest, 'gs'])
plt_vals.extend([Xtest, y_pred, 'bs'])

plt.plot(*plt_vals)
plt.legend(['train ','golden','pred'])
plt.show(block=False)

plt.figure()
plt_vals = []


plt_vals.extend([Xtest, ytest, 'gs'])
plt_vals.extend([Xtest, y_pred, 'bs'])

plt.plot(*plt_vals)
plt.legend(['golden','pred'])
plt.show(block=False)

#######################################################3333333
#### case with only 10 samples 



Xtrain=X[0:10,:]
ytrain=y[0:10,:]
Xtest=X[11:,:]
ytest=y[11:,:]





gp_2 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

gp_2.fit(Xtrain, ytrain)
lik, grad=gp_2.log_marginal_likelihood(theta=[1.0,1.0,1.0],  eval_gradient=True)

def obj_2(theta):
	a,b=gp_2.log_marginal_likelihood(theta, True)
	return -1*a,-1*b


n_iter=20
theta=np.zeros((3,n_iter))	
for i in range(n_iter):
	x0=np.random.uniform(1,10,3)
	theta[:,i], f, c=scipy.optimize.fmin_tnc(obj_2, x0)

## prediction

y_pred, sigma = gp_2.predict(Xtest, return_std=True)





# plots
plt.figure()
bins = np.linspace(-2, 5, 100)




plt.hist(theta[0,:],bins, edgecolor='k',label='alpha2')
plt.hist(theta[1,:],bins,edgecolor='k', label='l')
plt.hist(theta[2,:], bins,edgecolor='k', label='delta')
plt.legend(loc='upper right')
plt.show(block=False)









plt.figure()
plt_vals = []

plt_vals.extend([Xtrain, ytrain, 'ro'])
plt_vals.extend([Xtest, ytest, 'gs'])
plt_vals.extend([Xtest, y_pred, 'bs'])

plt.plot(*plt_vals)
plt.legend(['train ','golden','pred'])
plt.show(block=False)

plt.figure()
plt_vals = []


plt_vals.extend([Xtest, ytest, 'gs'])
plt_vals.extend([Xtest, y_pred, 'bs'])

plt.plot(*plt_vals)
plt.legend(['golden','pred'])
plt.show(block=False)















