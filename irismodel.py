# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 09:45:46 2021

@author: ab522tx
"""

import pandas as pd

import pickle

data = pd.read_csv("Iris.csv")

#Iris-versicolor    50   0
#Iris-virginica     50   1
#Iris-setosa        50   2

X = data.drop(['Id','Species'],axis=1)
Y = data.iloc[:,-1]

#from sklearn.model_selection import train_test_split
#x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state =0)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=9)

knn.fit(X,Y)

#print(knn.predict(np.array([5.1,3.5,1.4,0.2]).reshape(1,-1)))

# Saving model to disk
pickle.dump(knn,open('knn_model.pkl','wb'))

# Loading model to compare results
model = pickle.load(open('knn_model.pkl','rb'))
model.predict([[5.1,3.5,1.4,0.2]])





