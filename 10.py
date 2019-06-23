
# coding: utf-8

# In[57]:


from numpy import *


# In[58]:


import operator


# In[59]:


from os import listdir


# In[60]:


import matplotlib


# In[61]:


import matplotlib.pyplot as plt


# In[62]:


import pandas as pd


# In[63]:


import numpy.linalg


# In[64]:


from scipy.stats.stats import pearsonr


# In[65]:


def kernel(point,xmat,k):
    m,n= shape(xmat)
    weights=mat(eye((m)))
    for j in range(m):
     diff = point - X[j]
     weights[j,j]= exp(diff*diff.T/(-2*k**2))
    return weights    


# In[66]:


def localWeight(point,xmat,ymat,k):
    wei=kernel(point,xmat,k)
    W=(X.T*(wei*X)).I*(X.T*(wei*ymat.T))
    return W


# In[67]:


def localWeightRegression(xmat,ymat,k):
    m,n=shape(xmat)
    ypred=zeros(m)
    for i in range(m):
     ypred[i]=xmat[i]*localWeight(xmat[i],xmat,ymat,k) 
    return ypred


# In[68]:


#load data points
data=pd.read_csv('tips.csv')
bill=array(data.totbill)
tip=array(data.tip)


# In[69]:


#Preparing and add 1 in bill
mbill=mat(bill)
mtip=mat(tip)
m=shape(mbill)[1]
one=mat(ones(m))
X=hstack((one.T,mbill.T))


# In[70]:


#set k here
ypred=localWeightRegression(X,mtip,0.5)
SortIndex=X[:,1].argsort(0)
xsort=X[SortIndex][:,0]
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.scatter(bill,tip,color='green')
ax.plot(xsort[:,1],ypred[SortIndex],color='red',linewidth=5)
plt.xlabel('Total Bill')
plt.ylabel('Tip')
plt.show()

