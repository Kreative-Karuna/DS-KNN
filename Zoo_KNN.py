# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 14:11:09 2022

@author: Karuna Singh
"""

import pandas as pd
import numpy as np

animal = pd.read_csv(r"D:\\Data Science Files\\Datasets_360\\KNN\\Zoo.csv")

# EDA
animal.head() # checking top 5 records
animal.isna().sum() # Checking missing values
animal.info() # Checking data details
animal.describe() # Checking statistical dimensions of the data
animal.duplicated().sum() # checking for duplicate values


# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
animal_n = norm_func(animal.iloc[:, 1:]) 
animal_n.describe()

X = np.array(animal_n.iloc[:,:]) # Predictors 
Y = np.array(animal['type']) # Target 

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 21)
knn.fit(X_train, Y_train)

pred = knn.predict(X_test)
pred

# Evaluate the model
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, pred))
pd.crosstab(Y_test, pred, rownames = ['Actual'], colnames= ['Predictions']) 


# error on train data
pred_train = knn.predict(X_train)
print(accuracy_score(Y_train, pred_train))
pd.crosstab(Y_train, pred_train, rownames=['Actual'], colnames = ['Predictions']) 


# creating empty list variable 
acc = []

# running KNN algorithm for 3 to 50 nearest neighbours(odd numbers) and 
# storing the accuracy values

for i in range(3,50,2):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train, Y_train)
    train_acc = np.mean(neigh.predict(X_train) == Y_train)
    test_acc = np.mean(neigh.predict(X_test) == Y_test)
    acc.append([train_acc, test_acc])


import matplotlib.pyplot as plt # library to do visualizations 

# train accuracy plot 
plt.plot(np.arange(3,50,2),[i[0] for i in acc],"ro-")

# test accuracy plot
plt.plot(np.arange(3,50,2),[i[1] for i in acc],"bo-")


# from the plot it is evident that K=3 will give the best model

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)

pred = knn.predict(X_test)
pred
accuracy_score(Y_test, pred)
