# -*- coding: utf-8 -*-
"""
Created on Fri May 22 04:53:52 2020

@author: YUSUFSENYURT
"""


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np

regression = LinearRegression()
data = pd.read_csv("insurance.csv")

expenses = data.expenses.values.reshape(-1,1)
ageBmi = data[["age","bmi"]].values

regression.fit(ageBmi,expenses)

print(regression.predict([[20,20],[20,25]]))