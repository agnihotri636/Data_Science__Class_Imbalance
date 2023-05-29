#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Loading libraries

import pandas as pd
import imblearn
from sklearn.metrics import f1_score
import numpy as np
from sklearn.linear_model import LogisticRegression
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
from imblearn.under_sampling import NearMiss


# In[ ]:


#loading the data
data=pd.read_csv('Data for Problem 3/creditcard.csv')

#test train split
training_data = data.sample(frac=0.7, random_state=25)
testing_data = data.drop(training_data.index)

#genearting the test set , class and features separated
X_test=testing_data.drop(columns=['Class'],axis=1)
Y_test=testing_data[['Class']]


# In[ ]:


#checking if the class column is imbalanced
training_data.Class.value_counts()


# # upsampling

# In[ ]:


#upsampling the data
#create two different dataframe of majority and minority class 

df_majority = training_data[(training_data['Class']==0)] 
df_minority = training_data[(training_data['Class']==1)] 

# upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,    # sample with replacement
                                 n_samples= len(df_majority), # to match majority class
                                 random_state=42)  # reproducible results

# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_minority_upsampled, df_majority])


# In[ ]:


#checking class frequencies post upsampling the train data
df_upsampled.Class.value_counts()


# In[ ]:


#creating X_train and Y_train from the upsampled data

X_train=df_upsampled.drop(columns=['Class'],axis=1)
Y_train=df_upsampled[['Class']]


# In[ ]:


#Creating model

classifier = LogisticRegression()
classifier.fit(X_train,Y_train)
predictions = classifier.predict(X_test)
predictions=pd.DataFrame(predictions)


# In[ ]:


#checking fscore

f_score_upsampling=f1_score(Y_test, predictions)


# # balanced bagging

# In[ ]:


#Create an instance
classifier = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(),
                                sampling_strategy='not majority',
                                replacement=False,
                                random_state=42)
#creating tgrain data
X_train=training_data.drop(columns=['Class'],axis=1)
Y_train=training_data[['Class']]

classifier.fit(X_train, Y_train)
#preds = classifier.predict(X_test)

preds=classifier.predict(X_test)
f_score_balanced_bag=f1_score(Y_test, preds)


# # SMOTE

# In[ ]:


#Creating train data
X_train=training_data.drop(columns=['Class'],axis=1)
Y_train=training_data[['Class']]


# In[ ]:


Y_train.value_counts()


# In[ ]:


#initialising smote
sm = SMOTE(random_state = 2)
X_train_res, y_train_res = sm.fit_resample(X_train, Y_train)


# In[ ]:


y_train_res.value_counts()


# In[ ]:


#creating model
classifier = LogisticRegression()
classifier.fit(X_train_res, y_train_res)
predictions = classifier.predict(X_test)


# In[ ]:


#checking fscore
f_score_smote=f1_score(Y_test, predictions)


# In[ ]:


#F SCORE

print('---------F score----------','\n',
     'Upsampling : ',f_score_upsampling,'\n',
      'SMOTE : ',f_score_smote,'\n',
      'Balanced Bagging : ',f_score_balanced_bag
     )

