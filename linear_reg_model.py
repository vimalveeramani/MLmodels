# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 09:38:04 2023

@author: pc
"""

# required modules for analysing and visualization
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


#to download required data set
data = yf.download('SPY')
plt.figure(figsize=(10, 5))
data['Close'].plot()

#reshape the date
data = data.reset_index()
x = np.array(data.index).reshape(-1, 1)
y = data['Close']


#linear regression modelling
linreg = LinearRegression().fit(x, y)
linreg.score(x, y)
predictions = linreg.predict(x)

#plotting of data
plt.figure(figsize=(15,5))
plt.plot(data['Close'])
plt.plot(data.index, predictions)

#print score
print('R^2:', linreg.score(x, y))

