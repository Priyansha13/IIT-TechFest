#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install sklearn')


# In[2]:


import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt


# In[20]:


df = pd.read_csv('canada_per_capita_income.csv')
df


# In[21]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('year')
plt.ylabel('PCI')
plt.scatter(df.year,df.PCI,color='red',marker='+')


# In[23]:


new_df = df.drop('PCI',axis='columns')
new_df


# In[25]:


PCI = df.PCI
PCI


# In[26]:


reg = linear_model.LinearRegression()
reg.fit(new_df,PCI)


# In[27]:


reg.predict([[2020]])


# In[ ]:




