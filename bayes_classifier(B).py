import pandas as pd#importing pandas for importing csv data
import numpy as np#importing numpy for data manipulation
from sklearn.model_selection import train_test_split #importing train_test_split to split the given data
from sklearn.neighbors import KNeighborsClassifier #importing knn method from sklearn.neighbors
from sklearn.metrics import confusion_matrix,mean_squared_error, accuracy_score #importing confusion_matrix for computation of computation matrix
                                                            # and accuracy for calculating accuracy of prediction
from sklearn.preprocessing import MinMaxScaler, StandardScaler#Used to normalize the data
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression#Used for regression process
from matplotlib import style
from sklearn.preprocessing import PolynomialFeatures
plt.style.use('fivethirtyeight') #using style method 


# Question :- 1
print("Question :- 1")
df = pd.read_csv("abalone.csv")#Reading the csv file
X = df['Shell weight'].values.reshape(-1,1) 
y = df['Rings'].values.reshape(-1,1)
# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
regressor = LinearRegression()  #Setting up the linear regression model
regressor.fit(X_train, y_train) #training the algorithm
t_pred = regressor.predict(X_train) #For predicting the training data
y_pred = regressor.predict(X_test) #For predicting the testing data
plt.scatter(X_test, y_test, color = "#0099ff")#Plotting the best fitted of line
plt.title("Best fitted plot")
plt.plot(X_test, y_pred, color = "#78736f", linewidth = 6)
plt.ylabel("Rings")
plt.xlabel("Shell Weight")
plt.show()
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_train, t_pred))) #printing the RMSE for training and
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))#testing data

# Plotting the original and predicted Ring value
plt.scatter(y_test, y_pred)
plt.title("True vs Predicted(Rings)")
plt.ylabel("Rings(Predicted)")
plt.xlabel("Rings(True)")
plt.show()

# Function to plot the bar-plot
def barplot(dictt):
    plt.title("Bar-Plot for P vs RMSE")
    plt.bar(dictt.keys(), dictt.values(), color="#02a0e3", width = 0.5)
    plt.xlabel("----P--->")
    plt.ylabel("--RMSE-->")
    plt.xticks(list(dictt.keys()))
    plt.show()
    
# Question :- 2
print("\nQuestion:-2")
def multiple_lin_regression():    
    df = pd.read_csv("abalone.csv")#Getting the data and setting up the multiple(linear) regression model.
    X = df.drop('Rings', axis = 1)
    y = df['Rings'].values.reshape(-1,1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    regressor = LinearRegression()  
    regressor.fit(X_train, y_train)
    t_pred = regressor.predict(X_train) #predicting the trainig data
    regressor2 = LinearRegression()  
    regressor2.fit(X_test, y_test)
    y_pred = regressor2.predict(X_test) #prdicting the test data
    print('Root Mean Squared Error(for training set):', np.sqrt(mean_squared_error(y_train, t_pred)))#Printing the RMSE
    print('Root Mean Squared Error(for testing set):', np.sqrt(mean_squared_error(y_test, y_pred)))
    plt.scatter(y_test, y_pred) #plotting the true nad predicted Ring
    plt.title("True vs Predicted(Rings)")
    plt.ylabel("Rings(Predicted)")
    plt.xlabel("Rings(True)")
    plt.show()
multiple_lin_regression()

# Question :- 3
print("\nQuestion :-3")
def polynomial_regression(p, plot):
    df = pd.read_csv("abalone.csv") #Getting the data and setting up the polynomial(linear) regression model.
    X = df['Shell weight'].values.reshape(-1, 1)
    y = df['Rings'].values.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    poly_features = PolynomialFeatures(degree = p)
    x_poly = poly_features.fit_transform(X_train)#Transforming into the polynomial relation 
    x2_poly = poly_features.fit_transform(X_test)
    regressor = LinearRegression()
    regressor.fit(x_poly, y_train)
    t_pred = regressor.predict(x_poly)#predicting the training data
    y_pred = regressor.predict(x2_poly)#predicting the test data
    
    if(plot=="fitted"):#plotting the best fitted line and with RMSE.
        print('Root Mean Squared Error(for train):', np.sqrt(mean_squared_error(y_train, t_pred)))
        plt.title("Best fitted plot")
        plt.scatter(X_train, y_train)
        plt.scatter(X_train, t_pred, color = "#2b2b2b") 
        plt.ylabel("Rings")
        plt.xlabel("Shell Weight")
        plt.show()
        
    elif(plot=="rings"):#plotting the true and predicted value of Rings
        plt.scatter(y_test, y_pred, color = "gray")
        plt.title("True vs Predicted(Rings)")
        plt.ylabel("Rings(predicted)")
        plt.xlabel("Rings(test)")
        plt.show()
    else:
        #else returning the RMSE of training and test data.
        return mean_squared_error(t_pred, y_train, squared = False), mean_squared_error(y_pred, y_test, squared = False)

p_values = [2,3,4,5] #list of value for different value of degree
rmse_test = {i:0 for i in p_values}
rmse_train = {i:0 for i in p_values}

# Calling function for different value of degree in order to get RMSE 
for p in p_values:
    train, test = polynomial_regression(p,"null")
    rmse_test[p] = test
    rmse_train[p] = train
# Plotting the bar-plot of RMSE for different value of degree of both training and testing data
barplot(rmse_train)
barplot(rmse_test)
best_p = min(rmse_train, key = rmse_train.get)
polynomial_regression(best_p, "fitted")#plotting the best fiited line for the data

best_p = min(rmse_test, key = rmse_test.get)
polynomial_regression(best_p, "rings")#plotting the true and predicted Rings value
barplot(rmse_test)

# Question :- 4
print("\nQuestion:-4")
def multiple_polynomial_regression(p, plot):
    df = pd.read_csv("abalone.csv")#Getting the data and setting up the polynomial(linear) regression model.
    X = df.drop('Rings', axis = 1)
    y = df['Rings'].values.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    poly_features = PolynomialFeatures(degree = p) #p is the degree
    x_poly = poly_features.fit_transform(X_train)#Transforming into polynomial relation 
    x2_poly = poly_features.fit_transform(X_test)
    regressor = LinearRegression()
    regressor.fit(x_poly, y_train)
    t_pred = regressor.predict(x_poly)#predicting the training and testing data
    y_pred = regressor.predict(x2_poly)
    if(plot==True):#To plot the true and predicted Rings
        plt.scatter(y_test, y_pred, color="orange")
        plt.title("True vs Predicted(Rings)")
        plt.ylabel("Rings(predicted)")
        plt.xlabel("Rings(test)")
        plt.show()
    else:
        #else returning the RMSE
        return mean_squared_error(t_pred, y_train, squared = False), mean_squared_error(y_pred, y_test, squared = False)

# Calling function for different value of degree in order to get RMSE 
p_values = [2,3,4,5]
rmse_test = {i:0 for i in p_values}
rmse_train = {i:0 for i in p_values}
# Plotting the bar-plot of RMSE for different value of degree of both training and testing data
for p in p_values:
    train, test = multiple_polynomial_regression(p, plot = False)
    rmse_test[p] = test
    rmse_train[p] = train

#getting the degree which minimises the RMSE     
best_p = min(rmse_test, key = rmse_test.get)
# Plotting the bar-plot for that degree.
barplot(rmse_train)
barplot(rmse_test)
multiple_polynomial_regression(best_p, True)#plotting the true and predicted Rings value