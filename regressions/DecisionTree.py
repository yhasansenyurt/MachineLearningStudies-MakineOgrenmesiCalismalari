# -*- coding: utf-8 -*-
"""
Created on Sun May 24 03:58:25 2020

@author: YUSUFSENYURT
"""


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from textblob import TextBlob


data = pd.read_csv("positions.csv")

level = data.Level.values.reshape(-1,1)
salary = data.Salary.values.reshape(-1,1)

regression = DecisionTreeRegressor()
regression.fit(level,salary)

print(regression.predict([[8.3]]))

plt.scatter(level,salary,c="r")
# plt.plot(level,regression.predict(level),c="g")
x = np.arange(min(level),max(level),0.01).reshape(-1,1)
plt.plot(x,regression.predict(x),c="g")
plt.show()