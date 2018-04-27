from __future__ import division 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("faithful.csv") #load data into pandas dataframe

#split data
y = df.as_matrix(columns=['eruptions']).ravel() #

X = df.as_matrix(columns=['waiting']).ravel() #
X=np.expand_dims(X, axis=1)
#X = np.hstack((np.ones((X.shape[0],1)),X, X**2,X**3)) #
#X = np.hstack((np.ones((X.shape[0],1)),X))


X=X.astype('float')
y=np.expand_dims(y, axis=1)
data=X
n=X.shape[0]
d=X.shape[1]

#mxx=X.max(axis=0)
#Xn=X/mxx  

def kernel(x1, x2, alpha2, l):
    return alpha2*np.exp(-1 * (np.linalg.norm(x1-x2) ** 2) / (2*l**2))


def matrix(X1,X2,alpha2, l):
	n1=X1.shape[0]
	n2=X2.shape[0]
	m= np.zeros((n1, n2))
	for i in range (n1):
		for j in range (n2):
			m[i,j]=kernel(X1[i,:],X2[j,:],alpha2, l)
			
	return m


def cov(Xp,X,sigma2,alpha2, l):
	temp=np.dot(matrix(Xp,X,alpha2, l),np.linalg.inv((matrix(X,X,alpha2, l)+sigma2*np.eye(n))) )
	temp=np.dot(temp,matrix(X,Xp,alpha2, l)) 
	return matrix(Xp,Xp,sigma2, l)-temp


def meen(Xp,X,y,sigma2,alpha2, l):
	temp=np.dot(matrix(Xp,X,alpha2, l),np.linalg.inv((matrix(X,X,alpha2, l) +sigma2*np.eye(n)   )) )
	return np.dot(temp,y)

 

alpha2=1
sigma2=1
l=[1,5,10,100]




x=np.linspace(0,100,200)
Xp=np.expand_dims(x, axis=1)
#Xp = np.hstack((np.ones((Xp.shape[0],1)),Xp, Xp**2,Xp**3)) #
#Xp = np.hstack((np.ones((Xp.shape[0],1)),Xp)) 
#Xpn=Xp/mxx 


for i in range(len(l)):
	plt.figure()
	mu=meen(Xp,X,y,sigma2,alpha2, l[i])
	cc=cov(Xp,X,sigma2,alpha2, l[i])


	conf=np.diag(cc)
	conf=np.sqrt(conf)
	conf=np.expand_dims(conf, axis=1)

	y1=y
	y2=mu
	y3=mu-3*conf
	y4=mu+3*conf



	color=['r','b','g','c','m']
	plt_vals = []
	plt_vals.extend([X, y1, 'ro'])
	plt_vals.extend([x, y2, 'b-'])
	plt_vals.extend([x, y3, 'g-'])
	plt_vals.extend([x, y4, 'c-'])

	plt.plot(*plt_vals)
	plt.legend(['data','model','low','high'])
	plt.title('l='+str(l[i]))
	plt.show(block=False)




