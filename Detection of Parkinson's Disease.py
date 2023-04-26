#!/usr/bin/env python
# coding: utf-8

# In[140]:


#import necessary model
import numpy as p
import pandas as pd
import seaborn as sbn
from sklearn.preprocessing import StandardScaler as ss
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[141]:


#load dataset
df = pd.read_csv('pd_speech_features.csv')
df


# In[142]:


df.shape


# In[143]:


df.info()


# In[144]:


#remove null values
df.dropna()


# In[145]:


df.describe


# In[148]:


sbn.countplot(df['numPulses'])


# In[149]:


sbn.histplot(df['tqwt_kurtosisValue_dec_30'],kde=True)


# In[150]:


sbn.kdeplot(df['numPulses'])


# In[151]:


#load specific values in x and y
x=df.iloc[:, 0:754].values
y=df.iloc[:, -1].values


# In[152]:


x


# In[153]:


y


# In[154]:


#scaling data
x = ss().fit_transform(x)


# In[155]:


#splitting dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)


# In[158]:


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_train)


# In[159]:


x_train.shape


# In[160]:


x_test.shape


# In[161]:


#applying PCA on train and test data
pca = PCA(n_components = 6)
x_train_n = pca.fit_transform(x_train)
x_test_n = pca.transform(x_test)


# In[162]:


x_train_n.shape


# In[163]:


x_test_n.shape


# In[164]:


#applying Logistic Regression before PCA
lr = LogisticRegression(random_state = 42)
lr.fit(x_train, y_train)


# In[165]:


#applying Logitic Regression after PCA
lr1 = LogisticRegression(random_state = 42)
lr1.fit(x_train_n, y_train)


# In[166]:


#prediction on data
y_pred = lr.predict(x_test)


# In[167]:


y_pred_n = lr1.predict(x_test_n)


# In[168]:


#accuracy of model
accuracy_score(y_pred, y_pred_n)*100


# In[ ]:




