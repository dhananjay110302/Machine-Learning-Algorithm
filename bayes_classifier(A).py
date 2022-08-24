"""
Name : Amrendra Kumar
Roll NO : B20080
Lab :- 5(A)
Mobile No : 8840478495
Branch : CSE
"""

#Importing the different libraries in order to process the things.
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.mixture import GaussianMixture #importing the GaussianMixture for multi-model bayes classifiction  
from sklearn.metrics import accuracy_score

def traintestsplit(df):#Writing a function to split the data
    X = df.iloc[:, 0:-1].values
    y = df['Class'].values
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3,stratify=y, random_state=42,shuffle=True)
    return X_train, X_test, y_train, y_test #returning all the train, test splitted data

def prob(x, w, mean, cov):#Function to calculate the probability
    p = 0
    for i in range(len(w)):
#       From the Bayes's formula claculating the probability for that class 
        p += w[i] * scipy.stats.multivariate_normal.pdf(x, mean[i], cov[i], allow_singular=True)        
    return p

def gmmbayes(df,k):#Creating a function for setting up the GMM model for the given value of cluster(k)
    
    df_0=df[df['Class']==0] #separating the data for class = 0
    df_1=df[df['Class']==1] #separating the data for class = 1
    
#     Calculating the prior probability
    prior0=len(df_0)/(len(df_0)+len(df_1))
    prior1=1-prior0
    X_train0, X_test0, y_train0, y_test0 = traintestsplit(df_0)
    X_train1, X_test1, y_train1, y_test1 = traintestsplit(df_1)
    
    test = np.concatenate((X_test0, X_test1))
    pred = np.concatenate((y_test0, y_test1))
    
#     Setting up the GMM classifier Model
    gmm = GaussianMixture(n_components=k)
    gmm.fit(X_train0)    

#After our model has converged, the weights, means, and covariances should be solved! We can print them out.
    
    gmm2 = GaussianMixture(n_components=4)
    gmm2.fit(X_train1)
    
    ypred = []    
    for x in test:
        ypred.append(  0 if prob(x, gmm.weights_, gmm.means_, gmm.covariances_)*prior0\
                           > prob(x, gmm2.weights_, gmm2.means_, gmm2.covariances_)*prior1 else 1 )
    print("Accuracy for GMM  Bayes Classifier with ",k,"components")
    print(accuracy_score(pred, ypred)) #printing the accuracy.
    print(confusion_matrix(pred, ypred)) #printing the confusion matrix
    
    return accuracy_score(pred, ypred)

df = pd.read_csv('SteelPlateFaults-train.csv') #Reading the previously saved training data
l=[2,4,8,16] #calling the gmm function for different value of q(cluster).
acc_q = {i : 0 for i in l}
for i in l:
    acc_q[i] = gmmbayes(df,i)#saving the RMSE for different value of q.