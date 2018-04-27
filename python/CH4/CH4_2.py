from __future__ import division 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("faithful.csv") #load data into pandas dataframe

#split data
y = df.as_matrix(columns=['waiting']).ravel() #

X = df.as_matrix(columns=['eruptions']).ravel() #
X=np.expand_dims(X, axis=1)
X = np.hstack((np.ones((X.shape[0],1)),X, X**2,X**3)) #



X=X.astype('float')
y=np.expand_dims(y, axis=1)
data=X

x=np.linspace(1.3,5.5,20)
x=np.expand_dims(x, axis=1)
XT = np.hstack((np.ones((x.shape[0],1)),x, x**2,x**3)) #

n=X.shape[0]
d=X.shape[1]
K=np.eye(d)
A=np.eye(n)
mu=np.ones((d,1))
sig=1
mu0=0.5*np.ones((d,1))
X=X.T
XT=XT.T
mu_n= np.dot(np.dot(np.dot(XT.T,X), np.linalg.inv(np.dot(X.T,X)+sig*A)),y)
cov=np.dot(XT.T,XT)-np.dot(np.dot(np.dot(XT.T,X), np.linalg.inv(np.dot(X.T,X)+sig*A)),np.dot(X.T,XT) )

conf=np.diag(cov)
conf=np.sqrt(conf)
conf=np.expand_dims(conf, axis=1)
y1=y
y2=mu_n
y3=mu_n-3*conf
y4=mu_n+3*conf



color=['r','b','g','c','m']
plt_vals = []
plt_vals.extend([X[1,:], y1, 'ro'])
plt_vals.extend([x, y2, 'b-'])
plt_vals.extend([x, y3, 'g-'])
plt_vals.extend([x, y4, 'c-'])

plt.plot(*plt_vals)
plt.legend(['data','model','low','high'])
plt.show(block=False)





