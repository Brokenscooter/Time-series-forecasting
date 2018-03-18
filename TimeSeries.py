
# coding: utf-8

# In[1]:

import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import sklearn
from sklearn import preprocessing as pre
from statsmodels.tsa.arima_model import ARIMA


# In[2]:

df=pd.read_csv(r'C:\Users\<>\Desktop\MorepenLabs.csv', index_col= None)
df.head()


# In[3]:

df.count()


# In[4]:

df['Ticks'] = range(0,len(df.index.values))
df.head()


# In[5]:

df.plot(x='Ticks', y='Close', style='-', figsize=(20,10))
plt.show()


# In[6]:

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

fig, (ax, ax2) = plt.subplots(ncols=2, figsize = (20,10)) # create two subplots, one in each row

plot_acf(df['Close'], lags=50, ax=ax)
ax.set_title("Autocorrelation without detrending")
plot_pacf(df['Close'], lags=50, ax=ax2)
ax2.set_title("Partial Autocorrelation without detrending")

plt.show()


# In[7]:

#Detrending using differnce 1
series=df['Close']
X = series.values
diff = list()
for i in range(1, len(X)):
    value = X[i] - X[i - 1]
    diff.append(value)

plt.figure(figsize=(20,10))
plt.plot(diff)
plt.show()


# In[8]:

#Detrending with difference 2
series2=df['Close']
X2 = series.values
diff2 = list()
for i in range(1, len(X2)):
    value = X[i] - X[i - 2]
    diff2.append(value)

plt.figure(figsize=(20,10))
plt.plot(diff2)
plt.show()


# In[9]:

fig2, (ax, ax2) = plt.subplots(ncols=2, figsize = (20,10)) # create two subplots, one in each row

plot_acf(diff, lags=50, ax=ax)
ax.set_title("Autocorrelation with Difference 1")
plot_acf(diff2, lags=50, ax=ax2)
ax2.set_title("Autocorrelation with Difference 2")

plt.show()


# In[10]:

fig3, (ax, ax2) = plt.subplots(ncols=2, figsize = (20,10)) # create two subplots, two in a row

plot_pacf(diff, lags=50, ax=ax)
ax.set_title("Partial Autocorrelation with Difference 1")
plot_pacf(diff2, lags=50, ax=ax2)
ax2.set_title("Partial Autocorrelation with Difference 2")

plt.show()


# In[11]:

d=2
for p in range(3):
    for q in range(3):
        try:
            arima_mod=ARIMA(diff,(p,d,q)).fit(transparams=True)

            x=arima_mod.aic

            x1= p,d,q
            print('ARIMA :', x1,' AIC Value: ',x)

            aic.append(x)
            pdq.append(x1)
        except:
            pass


# In[12]:

#ARIMA

model1=ARIMA(diff, order=(0,2,1))
model1_fit=model1.fit(disp=0)
print(model1_fit.summary())


# In[13]:

d=2

for p in range(3):
    for q in range(3):
        try:
            arima_mod=ARIMA(diff2,(p,d,q)).fit(transparams=True)

            x=arima_mod.aic

            x1= p,d,q
            print ('ARIMA :', x1,' AIC Value: ',x)

            aic.append(x)
            pdq.append(x1)
        except:
            pass


# In[14]:

#ARIMA
model2=ARIMA(diff2, order=(2,2,1))
model2_fit=model2.fit(disp=0)
print(model2_fit.summary())


# In[15]:

residuals1=DataFrame(model1_fit.resid)
residuals2=DataFrame(model2_fit.resid)

fig4, (ax, ax2) = plt.subplots(ncols=2, figsize = (20,10)) # create two subplots, two in a row

residuals1.plot(ax=ax)
ax.set_title("Residual error with Difference 1, ARIMA(0,2,1)")
residuals2.plot(ax=ax2)
ax2.set_title("Residual error with Difference 2, ARIMA(2,2,1)")

plt.show()


# In[16]:

fig4, (ax, ax2) = plt.subplots(ncols=2, figsize = (20,10)) # create two subplots, two in a row

residuals1.plot(kind='kde', ax=ax)
ax.set_title('Diff 1, ARIMA (0,2,1)')
residuals2.plot(kind='kde', ax=ax2)
ax2.set_title('Diff 2, ARIMA (2,2,1)')

plt.show()


# In[17]:

print(residuals1.describe())


# In[18]:

print(residuals2.describe())


# In[22]:

from sklearn.metrics import mean_squared_error

X1= diff2
size = int(len(X1) * 0.66)
train1, test1 = X1[0:size], X1[size:len(X1)]
history1 = [x for x in train1]
predictions1 = list()
for t in range(len(test1)):
    model1 = ARIMA(history1, order=(2,2,1))
    model1_fit = model1.fit(disp=0)
    output1 = model1_fit.forecast()
    yhat1 = output1[0]
    predictions1.append(yhat1)
    obs1 = test1[t]
    history1.append(obs1)
    #print('predicted=%f, expected=%f' % (yhat, obs))


# In[42]:

X2= series2.values
size = int(len(X2) * 0.66)
train2, test2 = X2[0:size], X2[size:len(X2)]
history2 = [x for x in train2]
predictions2 = list()
for t in range(len(test2)):
    model2 = ARIMA(history2, order=(2,2,1))
    model2_fit = model2.fit(disp=0)
    output2 = model2_fit.forecast()
    yhat2 = output2[0]
    predictions2.append(yhat2)
    obs2 = test2[t]
    history2.append(obs2)
    #print('predicted=%f, expected=%f' % (yhat, obs))

    


# In[23]:

plt.figure(figsize=(20,10))
plt.plot(test1)
plt.plot(predictions1, color='red')

plt.show()


# In[55]:

plt.figure(figsize=(20,10))
plt.plot(test2)
plt.plot(predictions2, color='red')

plt.show()


# In[58]:

def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

forecast = model2_fit.forecast(steps=7)[0]
print(forecast)


# In[33]:

series2 =pd.read_csv(r'C:\Users\<>\Desktop\MP1.csv', header=0, parse_dates=[0], index_col=0, squeeze= True)

X= series2.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(2,2,1))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)


# In[34]:

plt.figure(figsize=(20,10))
plt.plot(test[1:150], label='Actual')
plt.plot(predictions[1:150], color='red', label='Predicted')
plt.legend()
plt.show()


# In[ ]:



