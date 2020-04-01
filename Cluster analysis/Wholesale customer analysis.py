#!/usr/bin/env python
# coding: utf-8

# In[17]:


# import statements
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from sklearn.cluster import KMeans


# In[18]:


df = pd.read_csv('Wholesale customers.csv')


# In[19]:


df.head(10)


# In[20]:


df.shape


# In[21]:


kmeans = KMeans(n_clusters=3).fit(df)
centroids = kmeans.cluster_centers_
print(centroids)


# In[23]:


df["kmeans"] = kmeans.fit_predict(df[df.columns[2:]])


# In[24]:


df.tail()


# In[25]:


plt.scatter(df['Channel'], df['Region'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.show()


# In[29]:


from sklearn.preprocessing import StandardScaler
features = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicatessen']
# Separating out the features
x = df.loc[:, features].values

# Standardizing the features
x = StandardScaler().fit_transform(x)






# In[34]:


from sklearn.decomposition import PCA
kmeans = PCA(n_components=2)

principalComponents = kmeans.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])


# In[35]:


principalDf.tail()


# In[38]:


print('Explained variation per principal component: {}'.format(kmeans.explained_variance_ratio_))


# In[39]:


# From the above output, you can observe that the principal component 1 holds 44.1% of the information while the principal component 2 holds 28% of the information.


# In[58]:


plt.figure()
plt.figure(figsize=(10,10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Principal Component - 1',fontsize=20)
plt.ylabel('Principal Component - 2',fontsize=20)
plt.title("Principal Component Analysis of Breast Cancer Dataset",fontsize=20)
targets = ['1', '2']
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = df['kmeans'] == target
    plt.scatter(principalDf.loc[indicesToKeep, 'principal component 1']
               , principalDf.loc[indicesToKeep, 'principal component 2'], c = color, s = 50)

plt.legend(targets,prop={'size': 15})


# In[53]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
data = principalComponents
#生成一个随机数据，样本大小为100, 特征数为2（这里因为要画二维图，所以就将特征设为2，至于三维怎么画？
#后续看有没有机会研究，当然你也可以试着降维到2维画图也行）
estimator = KMeans(n_clusters=2)#构造聚类器，构造一个聚类数为3的聚类器
estimator.fit(data)#聚类
label_pred = estimator.labels_ #获取聚类标签
centroids = estimator.cluster_centers_ #获取聚类中心
inertia = estimator.inertia_ # 获取聚类准则的总和
mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
#这里'or'代表中的'o'代表画圈，'r'代表颜色为红色，后面的依次类推
color = 0
j = 0 
for i in label_pred:
    plt.plot([data[j:j+1,0]], [data[j:j+1,1]], mark[i], markersize = 5)
    j +=1
plt.show()


# In[57]:





# In[ ]:





# In[ ]:




