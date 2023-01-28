#!/usr/bin/env python
# coding: utf-8

# # Let's Grow More (LGMVIP) - "DATA SCIENCE INTERN"
# Jan-2023
# AUTHOR - MUTHYALA NAGA RAJU
# 
# 
# 
# 
# 
# BEGINNER LEVEL TASK
# 
# 
# 
# 
# 
# TASK-1- Iris Flowers Classification ML Project :
# 
# 
# 
# 
# This particular ML project is usually referred to as the “Hello World” of Machine Learning. The iris flowers dataset contains numeric attributes, and it is perfect for beginners to learn about supervised ML algorithms, mainly how to load and handle data. Also, since this is a small dataset, it can easily fit in memory without requiring special transformations or scaling capabilities.

# # Dataset link :http://archive.ics.uci.edu/ml/machine-learning-databases/iris

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# %matplotlib inline
# import seaborn as sns
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC,LinearSVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score, plot_confusion_matrix, classification_report,confusion_matrix

# In[3]:


df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                names=["Sepal_Length_in_cm","Sepal_Width_in_cm","Petal_Length_in_cm","Petal_Width_in_cm","Species_Flower"])


# In[4]:


df


# In[5]:


df.head(10)


# In[6]:


df.shape


# In[7]:


df.columns


# In[8]:


df.info()


# In[9]:


df.isnull().sum()


# # Checking correlation

# In[11]:


df.describe()


# In[13]:


sns.pairplot(df,hue="Species_Flower")


# In[41]:


fig,(ax1,ax2)=plt.subplots(ncols=2,figsize=(16,5))
sns.scatterplot(x='Sepal_Length_in_cm',y='Petal_Length_in_cm',data=df,hue='Species_Flower',ax=ax1)
sns.scatterplot(x='Sepal_Width_in_cm',y='Petal_Width_in_cm',data=df,hue='Species_Flower',ax=ax2)


# In[15]:


plt.figure(figsize=(16,4))
plt.subplot(1,4,1)
sns.boxplot(data=df,y='Sepal_Length_in_cm')
plt.subplot(1,4,2)
sns.boxplot(data=df,y='Sepal_Width_in_cm',color='red')
plt.subplot(1,4,3)
sns.boxplot(data=df,y='Petal_Width_in_cm',color='orange')
plt.subplot(1,4,4)
sns.boxplot(data=df,y='Petal_Width_in_cm',color='cyan')


# In[16]:


sns.violinplot(y='Species_Flower', x='Sepal_Length_in_cm', data=df, inner='quartile')
plt.show()
sns.violinplot(y='Species_Flower', x='Sepal_Width_in_cm', data=df, inner='quartile')
plt.show()
sns.violinplot(y='Species_Flower', x='Petal_Length_in_cm', data=df, inner='quartile')
plt.show()
sns.violinplot(y='Species_Flower', x='Petal_Width_in_cm', data=df, inner='quartile')
plt.show()


# In[17]:


plt.figure(figsize=(7,5))
sns.heatmap(df.corr(), annot=True,cmap='coolwarm')
plt.show()


# # Building Model , Training and Testing

# In[18]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[19]:


df['Species_Flower'] = le.fit_transform = (df['Species_Flower'])
df.head(10)


# In[20]:


from sklearn.model_selection import train_test_split
X = df.drop(columns=['Species_Flower'])
Y = df['Species_Flower']
x_train , x_test , y_train , y_test = train_test_split(X , Y , test_size = 0.3)


# # 1. Logistic Regression

# In[21]:


# Initialize a Logistic Regression
lg= LogisticRegression(max_iter=1000)


# In[22]:


lg.fit(x_train,y_train)


# In[23]:


# Predict on the test set and calculate accuracy
y_pred=lg.predict(x_test)
score=accuracy_score(y_test,y_pred)


# In[27]:


NB= MultinomialNB()


# In[28]:


NB.fit(x_train,y_train)

