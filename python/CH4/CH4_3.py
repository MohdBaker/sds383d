from __future__ import division 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#def kernel(d,sigma2, l):
 #   f=np.zeros(d.shape)
 #   for i in range(d.shape[0]):
 #   	f[i]=-(np.linalg.norm(d[i])**2/(2*l**2))
 #   return sigma2*np.exp(f) 


def kernel(x1, x2, sigma2, l):
    return sigma2*np.exp(-1 * (np.linalg.norm(x1-x2) ** 2) / (2*l**2))

def matrix(d,sigma2, l):
    return [[kernel(x1,x2,sigma2, l) for x2 in d] for x1 in d]





d=np.linspace(0,100,200)
sigma2=2
l1=1
l2=10
l3=0.1
f1=matrix(d,sigma2, l1)
f2=matrix(d,sigma2, l2)
f3=matrix(d,sigma2, l3)

mean = [0 for x in d]

color=['r','b','g','c','m']
sigma2=[1,2,3,4,5]
plt.figure()
plt_vals = []
for i in range(0, 5):
    ys = np.random.multivariate_normal(mean, f1)
    plt_vals.extend([d, ys, color[i]])
plt.plot(*plt_vals)
plt.title('l=1')
plt.show(block=False)

plt.figure()
plt_vals = []
for i in range(0, 5):
    ys = np.random.multivariate_normal(mean, f2)
    plt_vals.extend([d, ys, color[i]])
plt.plot(*plt_vals)
plt.title('l=10')
plt.show(block=False)

plt.figure()
plt_vals = []
for i in range(0, 5):
    ys = np.random.multivariate_normal(mean, f3)
    plt_vals.extend([d, ys, color[i]])
plt.plot(*plt_vals)
plt.title('l=0.1')
plt.show(block=False)


plt.figure()
plt_vals = []
ys = np.random.multivariate_normal(mean, f1)   	
plt_vals.extend([d, ys, color[0]])
ys = np.random.multivariate_normal(mean, f2)   	
plt_vals.extend([d, ys, color[1]])
ys = np.random.multivariate_normal(mean, f3)   	
plt_vals.extend([d, ys, color[2]])

plt.plot(*plt_vals)
plt.legend(['l=1','l=10','l=0.1'])
plt.title('Comparison')
plt.show(block=False)

   


