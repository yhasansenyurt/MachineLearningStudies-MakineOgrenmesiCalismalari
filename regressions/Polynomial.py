# -*- coding: utf-8 -*-
"""
Created on Fri May 22 05:09:50 2020

@author: YUSUFSENYURT
"""


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import numpy as np

data = pd.read_csv("positions.csv")

x = data.Level.values.reshape(-1,1)
y = data.Salary.values.reshape(-1,1)

regression = LinearRegression()
### Linear Regression
regression.fit(x,y)
print(regression.predict([[8]]))

### Polynomial Regression

regression_poly = PolynomialFeatures(degree = 4)

x_poly = regression_poly.fit_transform(x)
regression2 = LinearRegression()

regression2.fit(x_poly,y)

print(regression2.predict(regression_poly.transform([[8.3]])))

plt.scatter(x,y)
plt.plot(x,regression.predict(x),color="r")
plt.plot(x,regression2.predict(x_poly),color = "g")
plt.show()