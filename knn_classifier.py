"""
Name : Amrendra Kumar
Roll NO : B20080
Mobile No : 8840478495
Branch : CSE

"""
import pandas as pd#importing pandas for importing csv data
import numpy as np#importing numpy for data manipulation
from sklearn.model_selection import train_test_split #importing train_test_split to split the given data
from sklearn.neighbors import KNeighborsClassifier #importing knn method from sklearn.neighbors
from sklearn.metrics import confusion_matrix, accuracy_score #importing confusion_matrix for computation of computation matrix
                                                            # and accuracy for calculating accuracy of prediction
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Question :- 1
print("Question :- 1")
df = pd.read_csv("SteelPlateFaults-2class.csv")
df.drop(['X_Minimum', 'Y_Minimum', 'TypeOfSteel_A300'], axis=True, inplace = True)#Dropping the required attribute
y = df.Class #taking out the label for classification 
x = df.drop('Class' , axis=1) #droping the 'Class' label for spliting the data 

#spliting the given datasets with respect to 'Class' label
x_train, x_test, x_label_train, x_label_test = train_test_split(x, y, test_size=0.3, random_state=42, shuffle=True)
train = pd.concat([x_train, x_label_train], axis = 1)
test = pd.concat([x_test, x_label_test], axis = 1)
# train.to_csv("SteelPlateFaults-test.csv") #Saving the train and test data
# test.to_csv("SteelPlateFaults-train.csv")
for k in [1, 3, 5]:#
    knn = KNeighborsClassifier(n_neighbors = k) #Fitting the data into KNN model
    knn.fit(x_train, x_label_train)
    y_pred = knn.predict(x_test)
    print("    *Confusion Matrix for k =",k , "is : \n", confusion_matrix(x_label_test, y_pred)) #Printing confusion matrix
    print("    *Accuracy for k = {0} comes out to be {1}.\n".format(k, 100*accuracy_score(x_label_test, y_pred))) #Printing accuracy


# Question :- 2)
print("Question :- 2")
min_max_data = pd.read_csv("SteelPlateFaults-2class.csv") #Reading file
min_max_data.drop(['X_Minimum', 'Y_Minimum', 'TypeOfSteel_A300'], axis=True, inplace = True) #Dropping the required attribute
scaler = MinMaxScaler() 
scaler.fit(min_max_data) #Normalizing teh given data
min_max_data = pd.DataFrame(scaler.transform(min_max_data), columns = min_max_data.columns)
y = min_max_data.Class #taking out the label for classification 
X = min_max_data.drop('Class', axis=1) #droping the 'Class' label for spliting the data 

#spliting the given datasets with respect to 'Class' label
x_train, x_test, x_label_train, x_label_test = train_test_split(x, y, test_size=0.3, random_state=42, shuffle=True)
train = pd.concat([x_train, x_label_train], axis = 1) #Constructing training data
test = pd.concat([x_test, x_label_test], axis = 1) #Constructing test data

# test.to_csv("SteelPlateFaults-test-Normalized.csv")  #Saving test normalized data
# train.to_csv("SteelPlateFaults-train-Normalized.csv") #Saving train normalized data

for k in [1, 3, 5]:#
    knn = KNeighborsClassifier(n_neighbors = k) #Fitting into KNN function with repective value of k.
    knn.fit(x_train, x_label_train)
    y_pred = knn.predict(x_test)
    print("    *Confusion Matrix for k =",k , "is : \n", confusion_matrix(x_label_test, y_pred)) #For constructing confusion matrix
    print("    *Accuracy for k = {0} comes out to be {1}.\n".format(k, 100*accuracy_score(x_label_test, y_pred))) #For finding accuracy
    
    
# Question :- 3
print("Question :- 3")
train0 = train[train["Class"] == 0]
train1 = train[train["Class"] == 1]                         #splitting the train data based on the target variables
xtrain0 = train0[train0.columns[:-1]]
xtrain1 = train1[train1.columns[:-1]]

cov0 = np.cov(xtrain0.T)
cov1 = np.cov(xtrain1.T)                                    #covariance matrices for the train subsets

mean0 = np.mean(xtrain0)
mean1 = np.mean(xtrain1)

def likelihood(x, m, cov):                                  #likelihood function based on the bayes model
    ex = np.exp(-0.5*np.dot(np.dot((x-m).T, np.linalg.inv(cov)), (x-m)))
    return(ex/((2*np.pi)**5 * (np.linalg.det(cov))**0.5))

prior0 = len(train0)/len(train)                             #Calculating prior probability
prior1 = len(train1)/len(train)

predict = []
for i, x in x_test.iterrows():                              #classifying based on maximum likelihood
    p0 = likelihood(x, mean0, cov0) * prior0
    p1 = likelihood(x, mean1, cov1) * prior1
    if p0 > p1:
        predict.append(0)
    else:
        predict.append(1)
print("    *Confusion Matrix => \n", confusion_matrix(x_label_test, predict))          #confusion matrix
print("    *Accuracy Score => ", accuracy_score(x_label_test, predict))                #accuracy score
train0.cov().to_csv("Cov_Matrix(class_0).csv") #Saving the Covariance Matrix data both classes
train1.cov().to_csv("Cov_Matrix(class_1).csv")
#print(mean0)
#print(mean1)