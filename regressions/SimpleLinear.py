# -*- coding: utf-8 -*-
"""
Created on Thu May 21 18:52:25 2020

@author: YUSUFSENYURT
"""


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np


data = pd.read_csv("hw_25000.csv")
regression = LinearRegression()



height1 = data.Height.values.reshape(-1,1)
weight1 = data.Weight.values.reshape(-1,1)

regression.fit(height1,weight1)
print(regression.predict([[74]]))

x = np.arange(min(height1),max(height1)).reshape(-1,1)

plt.scatter(data["Height"],data["Weight"])
plt.plot(x,regression.predict(x),color="r")
plt.show()

print("Doğruluk payı =",r2_score(weight1,regression.predict(height1)))