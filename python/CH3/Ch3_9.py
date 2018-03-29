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

fit=sm.sampling(data=DATA, chains=3, iter=2000, warmup=1000, thin=10)

fig=fit.plot(['beta1','beta2'])

plt.show(fig)



