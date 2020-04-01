#!/usr/bin/env python
# coding: utf-8

# In[1]:


#This code is used to analysis customers, the data set includes three features: time to last consuming. frequency of consuming,
# total money of spending. I am going to use K-means to cluster customers.


# In[2]:


import pandas as pd


# In[3]:


#imput data
inputfile = 'consumption_data.xls'


# In[4]:


outputfile = 'data_type.xls'


# In[5]:


k = 3  # cluster numbers
iteration = 500  # iteration times
data = pd.read_excel(inputfile,index_col='Id') 


# In[6]:


R is time to last consuming. F is frequency of consuming, M is total money of spending
data[:3]


# In[7]:


#standalize dataset
data_zs = 1.0 * (data - data.mean())/data.std()


# In[8]:


from sklearn.cluster import KMeans 


# In[10]:


# Init the model, 3 clusters, 500 times
model = KMeans(n_clusters=k,n_jobs=4,max_iter=iteration)


# In[11]:


#Beigin Clustering
model.fit(data_zs)


# In[12]:


r = pd.concat([data,pd.Series(model.labels_,index=data.index)],axis=1)  # 将分类结果model.labels_打到数据data上去,axis＝1代表横向连接
r.columns=list(data.columns)+[u'cluster']  # 给新加的列赋一个列名


# In[13]:


r[:3]


# In[14]:


# Output to excel
r.to_excel(outputfile) 


# In[ ]:




