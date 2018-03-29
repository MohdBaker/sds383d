from __future__ import division 
import numpy as np
import pandas as pd
import pystan 
import matplotlib.pyplot as plt
import math

df = pd.read_csv("tea_discipline_oss.csv") #load data into pandas dataframe

#split data
y_l_0 = df.as_matrix(columns=['ACTIONS']).ravel() #

X_0 = df.as_matrix(columns=['GRADE']).ravel() #



# Remove -99
valid=np.where(y_l_0>0)
y_l=y_l_0[valid]
X=X_0[valid]
X=X.astype('float')
X=X  # get GRADE^2
y_l=y_l.astype('float')
y_l=np.expand_dims(y_l, axis=1)
X=np.expand_dims(X, axis=1)
### add intercept
X = np.hstack((np.ones((X.shape[0],1)).astype('int'),X)) #


X = np.hstack((X,y_l)) 


N=y_l.shape[0]
X=X.astype('int')
DATA=pd.DataFrame(X, columns=['intercept','x','y'])
#DATA=pd.DataFrame(X, columns=['x','y'])

DATA=DATA.to_dict('list')
DATA['N']=N
#DATA['intercept']=1


sm=pystan.StanModel(file='poisson.stan')

#fit=sm.sampling(data=DATA, chains=3, iter=1, warmup=1000, thin=10)

#fit.plot('beta')

#plt.show()

################# Censored data

# change -99 to 0
cens=np.where(y_l_0<0)
y_ls=np.zeros((cens[0].shape[0],1))
Xs=X_0[cens]
remov=np.where(Xs!='EE')
Xs=Xs[remov]
y_ls=y_ls[remov]
Xs=Xs.astype('float')
Xs=np.expand_dims(Xs, axis=1)
### add intercept
Xs = np.hstack((np.ones((Xs.shape[0],1)).astype('int'),Xs)) #


Xs = np.hstack((Xs,y_ls)) 


Nc=y_ls.shape[0]


Xc=Xs.astype('int')
DATA2=pd.DataFrame(Xc, columns=['interceptc','xc','yc'])
DATA2=DATA2.to_dict('list')
DATA2['Nc']=Nc
DATA2['U']=4
DATA_all=DATA.copy()
DATA_all.update(DATA2)

sm_cens=pystan.StanModel(file='poisson_cens.stan')
fit_cens=sm_cens.sampling(data=DATA_all, chains=3, iter=2000, warmup=1000, thin=10)

fig=fit_cens.plot(['beta1','beta2'])

plt.show(fig)


