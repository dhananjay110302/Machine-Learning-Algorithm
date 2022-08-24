# Importing all the needed libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
from math import sqrt, ceil
from sklearn.metrics import mean_absolute_percentage_error
from matplotlib import style
import statsmodels.api as sm

df = pd.read_csv("daily_covid_cases.csv") #Reading files
plt.style.use('fivethirtyeight') #using style method 

series = pd.read_csv('daily_covid_cases.csv',
parse_dates=['Date'],
index_col=['Date'],
sep=',')
series.dropna(inplace = True)
test_size = 0.35 # 35% for testing
X = series.values
tst_sz = ceil(len(X)*test_size)
train, test = X[:len(X)-tst_sz], X[len(X)-tst_sz:] #Spliting the series in train test ration as 0.65 : 0.35

def p_lag(lag, coeff): #Modeling the auto-regressive model for predicting the values
    window = lag
    model = AutoReg(train, lags=lag).fit()
    coef = model.params.tolist()
    if(coeff==True):
        print("Coefficient for lag = {0} is {1}".format(lag, coef))
    history = train[len(train)-window:]
    history = [history[i] for i in range(len(history))]
    predictions = list() # List to hold the predictions, 1 step at a time
    for t in range(len(test)):
        length = len(history)
        lag = [history[i] for i in range(length-window,length)]
        yhat = coef[0] # Initialize to w0
        for d in range(window):
            yhat += coef[d+1] * lag[window-d-1] # Add other values
            obs = test[t]
        predictions.append(yhat) #Append predictions to compute RMSE later
        history.append(obs)
    return predictions

def mape(test, pred): #Function to calculate Absolute Perecentag Error
    return np.sum(abs(np.array(test) - np.array(pred))/(np.array(test))) *100/215

def rmse(test, pred): #Function to calculate RMSE
    return np.sqrt(mean_squared_error(test, pred))*100/(sum(test)/215)[0]

def comp_lag(p): #Function to calculate autocorrelation
    a = sm.tsa.acf(df['new_cases'], nlags = p)
    return a

def barplot(dictt, Type): #Function to plot Bar plot 
    y_label = "---" + Type + "--->"
    plt.title("Bar-Plot for P vs RMSE")
    plt.bar(dictt.keys(), dictt.values(), color="#02a0e3", width = 0.5)
    plt.xlabel("----P--->")
    plt.ylabel(y_label)
    plt.xticks(list(dictt.keys()))
    plt.show()
    
    
print("Question :- 1")
plt.plot(df['Date'], df['new_cases'])
plt.ylabel("New Cases")
plt.xlabel("Date")
plt.show()
lag1 = comp_lag(1)
print("Autocorrelation between the given and one day lag data is {0}".format(lag1[1]))#printing the auto correlation

# Plotting the scatter plpot 
plt.scatter(df["new_cases"][1:], df['new_cases'][:-1], alpha = 0.35)
plt.xlabel("Given Time")
plt.ylabel("1 day lagged time")
plt.show()

# Plotting the Autocorrelation vs Lag Values
a = {i : comp_lag(i)[i] for i in range(1, 7)}
plt.plot(a.keys(), a.values(), color = "r")
plt.ylabel("Correlation")
plt.xlabel("Lag values")
plt.plot()

# Plotting the Auto correlation curve
plot_acf(df['new_cases'])
plt.xlabel("Lag value")
plt.ylabel("Correlation")
plt.show()

print("\nQuestion :- 2")
pred = p_lag(5, True)
print("RMSE for lagged 5 time series data is {0}".format(rmse(test, pred)))
print("MAPE for lagged 5 time series data is {0}".format(mape(test, pred)))

# plotting the scatter and line plot for predicting and true test data
plt.scatter(test, pred)
plt.xlabel("Test Data")
plt.ylabel("Predicted Data")
plt.show()

plt.plot(test)
plt.plot(pred)
plt.xlabel("Test Data")
plt.ylabel("Predicted Data")
plt.show()


print("\nQuestion :- 3")
# Calculating the RMSE and MAPE for different value of lag
l = [1, 5, 10, 15, 25]
Rmse = {i:0 for i in l}
Mape = {i : 0 for i in l}
for i in l:
    pred = p_lag(i, False)
    Rmse[i] = rmse(test, pred)
    Mape[i] = mape(test, pred)
print("RMSE for lag in {0} is {1}.".format(l, Rmse))
print("MAPE for lag in {0} is {1}.".format(l, Mape))

# plotting the bar plot for RMSE and MAPE
barplot(Rmse, "RMSE")
barplot(Mape, "Mape")


print("\nQuestion :- 4")
# Calculating the hueristic value
x = df['new_cases']
a = sm.tsa.acf(x, nlags=100)
lag = 0
h = 2/ sqrt(397)
for i in range(len(a)):
    if(a[i]<h):
        lag = i
        break
pred = p_lag(lag, False)

# Printing the RMSE and MAPE value for that hueristic value
print("RMSE for lagged {1} time series data is {0}".format(rmse(test, pred), lag))
print("MAPE for lagged {1} time series data is {0}".format(mape(test, pred), lag))