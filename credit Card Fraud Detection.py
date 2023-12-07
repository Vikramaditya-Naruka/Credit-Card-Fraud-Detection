#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[4]:


# loading the dataset to a Pandas DataFrame
credit_card_data = pd.read_csv('C:\\Users\\welcome\\Downloads\\DataFrame\\creditcard.csv')


# In[5]:


credit_card_data.head()


# In[6]:


credit_card_data.tail()


# In[7]:


# dataset informations
credit_card_data.info()


# In[8]:


# checking the number of missing values in each column
credit_card_data.isnull().sum()


# In[9]:


# distribution of legit transactions & fraudulent transactions
credit_card_data['Class'].value_counts()


# # This Dataset is highly unblanced
# 
# 0 --> Normal Transaction
# 
# 1 --> fraudulent transaction
# 
# 

# In[10]:


# separating the data for analysis
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]


# In[11]:


print(legit.shape)
print(fraud.shape)


# In[12]:


# statistical measures of the data
legit.Amount.describe()


# In[13]:


fraud.Amount.describe()


# In[14]:


# compare the values for both transactions
credit_card_data.groupby('Class').mean()


# # Under-Sampling
# 
# Build a sample dataset containing similar distribution of normal transactions and Fraudulent Transactions
# 
# Number of Fraudulent Transactions --> 492

# In[15]:


legit_sample = legit.sample(n=492)


# # Concatenating two DataFrames

# In[17]:


new_dataset = pd.concat([legit_sample, fraud], axis=0)


# In[18]:


new_dataset.head()


# In[19]:


new_dataset.tail()


# In[20]:


new_dataset['Class'].value_counts()


# In[22]:


new_dataset.groupby('Class').mean()


# # Splitting the data into Features & Targets

# In[23]:


X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']


# In[25]:


print(X)


# In[26]:


print(Y)


# # Split the data into Training data & Testing Data

# In[28]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)


# In[29]:


print(X.shape, X_train.shape, X_test.shape)


# # Model training

# # Logistic Regression

# In[30]:


model = LogisticRegression()


# In[31]:


# training the Logistic Regression Model with Training Data
model.fit(X_train, Y_train)


# # Model Evaluation

# # Acccuracy Score

# In[32]:


# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[33]:


print('Accuracy on Training data : ', training_data_accuracy)


# In[34]:


# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[35]:


print('Accuracy score on Test Data : ', test_data_accuracy)


# In[ ]:




