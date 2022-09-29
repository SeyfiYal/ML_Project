#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree  import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# In[3]:


df = pd.read_csv("heart.csv")


# ## ---EDA---

# In[58]:


df.head()


# In[4]:


### Check if there is missing datas ###

df.isnull().sum()


# In[36]:


### Rows and columns ###

df.shape


# In[37]:


df.info


# In[59]:


df.info()


# ### Hearth Disease

# In[65]:


################
## Add LABELS ##
################
plt.figure(figsize=(12, 7))
heartDisease_countplot = sns.countplot(x=df.HeartDisease,palette=["orange","Blue"])
heartDisease_countplot.set_title("Distribution of Target 'Heart Disease'")
heartDisease_countplot.set_xticklabels(['No', 'Yes'], fontsize=20)


# In[64]:


################
## Add LABELS ##
################
sns.countplot(data=df,x='Sex' , palette=['orange','Blue']);
plt.title('Sex Distribution')
plt.xlabel('Sex')
plt.ylabel('Count')


# In[75]:


ChestPainType_label = ['ATA', 'NAP', 'ASY', 'TA']
ChestPainType_size = [173, 203, 496, 46]
ChestPainType_explode = (0, 0, 0, 0)

plt.figure(figsize=(8,9))
plt.bar(ChestPainType_label, ChestPainType_size, color=['orange', 'blue', 'slateblue','red'])
plt.title("Count of ChestPainType")
plt.legend()

ax = plt.gca()
n_points = df.shape[0]
for i, p in enumerate(ax.patches):
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.annotate('{:.0%}'.format(height/n_points), (x, y + height + 0.01))


# ### RestingECG

# In[78]:



RestingECG_label = ['Normal', 'ST', 'LVH']
RestingECG_size = [552, 178, 188]
RestingECG_explode = (0, 0, 0)

plt.figure(figsize=(8,9))
plt.bar(RestingECG_label, RestingECG_size, color=['lime', 'red', 'slateblue'])
plt.title("Count of RestingECG")
plt.legend()


ax = plt.gca()
n_points = df.shape[0]
for i, p in enumerate(ax.patches):
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.annotate('{:.0%}'.format(height/n_points), (x, y + height + 0.01))


# ### ExeciseAngina

# In[81]:


############################
#####gives wrong output ####
#############################


ExerciseAngina_label = ['yes', 'no']
ExerciseAngina_size = [552, 178]
ExerciseAngina_explode = (0, 0)

plt.figure(figsize=(8,9))
plt.bar(ExerciseAngina_label, ExerciseAngina_size, color=['orange', 'blue'])
plt.title("ExersieAngina")
plt.legend()

# Add percentages to the bars
ax = plt.gca()
n_points = df.shape[0]
for i, p in enumerate(ax.patches):
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.annotate('{:.0%}'.format(height/n_points), (x, y + height + 0.01))


# ### ST_Slopea

# In[79]:


ST_Slope_label = ['Flat','Up','Down']
ST_Slope_size = [552, 178, 188]
ST_Slope_explode = (0, 0, 0)

plt.figure(figsize=(8,9))
plt.bar(ST_Slope_label, ST_Slope_size, color=['orange', 'blue', 'red'])
plt.title("Count of RestingECG")
plt.legend()

# Add percentages to the bars
ax = plt.gca()
n_points = df.shape[0]
for i, p in enumerate(ax.patches):
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.annotate('{:.0%}'.format(height/n_points), (x, y + height + 0.01))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




