from __future__ import division 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import wishart,multivariate_normal
from sklearn.preprocessing import scale


K=100

alpha=np.ones((K,4))
alpha[:,0]=10*alpha[:,0]
alpha[:,2]=0.1*alpha[:,2]
alpha[:,3]=0.01*alpha[:,3]

Z=np.zeros((5,K,4))

for j in range(4):
	for i in range(5):
		pii=np.random.dirichlet(alpha[:,j].squeeze())
		Z[i,:,j]=np.random.multinomial(10,pii)
 
plt_vals = []
color=['ro','bo','g*','c*','m*']	
for j in range(4):
	z=Z[:,:,j].squeeze()
	labels=np.sum(z, axis=0)
	plt_vals.extend([labels, color[j]])

plt.plot(*plt_vals)
plt.legend(['alpha=10','alpha=1','alpha=0.1','alpha=0.01'])
plt.show(block=False)





