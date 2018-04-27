from __future__ import division 
import numpy as np
import pandas as pd
import pystan 
import matplotlib.pyplot as plt
import math



df = pd.read_csv("faithful.csv") #load data into pandas dataframe

#split data
y = df.as_matrix(columns=['eruptions']).ravel() #

X = df.as_matrix(columns=['waiting']).ravel() #
X=np.expand_dims(X, axis=1)

X=X.astype('float')
y=np.expand_dims(y, axis=1)
data=X
n=X.shape[0]
d=X.shape[1]

Xtrain=X[0:int(0.75*n),:]
ytrain=y[0:int(0.75*n),:]
Xtest=X[int(0.75*n)+1:,:]
ytest=y[int(0.75*n)+1:,:]



N=Xtrain.shape[0]
N_predict=Xtest.shape[0]


DATA=pd.DataFrame(Xtrain, columns=['x'])
DATA=DATA.to_dict('list')
DATA['N']=N
DATA['N_predict']=N_predict
DATA['y']=ytrain.squeeze()
DATA['x_predict']=Xtest.squeeze()

sm=pystan.StanModel(file='gp_regression.stan')


fit=sm.sampling(data=DATA, chains=3, iter=2000, warmup=1000, thin=10)

params=fit.extract()

bins = np.linspace(-2, 5, 100)
 

log_alpha=np.log(params['alpha'])
log_rho=np.log(params['rho'])
log_sigma=np.log(params['sigma'])



plt.hist(log_alpha,bins, edgecolor='k',label='alpha')
plt.hist(log_sigma,bins,edgecolor='k', label='sigma')
plt.hist(log_rho, bins,edgecolor='k', label='rho')
plt.legend(loc='upper right')
plt.show()





### 
probs=[0.1,0.5,0.9]



#### apply the model 













