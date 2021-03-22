# -*- coding: utf-8 -*-
"""
Created on Sun May 24 04:15:58 2020

@author: YUSUFSENYURT
"""

import pandas as pd

from sklearn.ensemble import RandomForestRegressor


data = pd.read_csv("positions.csv")

level = data.iloc[:,1].values.reshape(-1,1)
salary = data.iloc[:,2].values

regression = RandomForestRegressor(n_estimators = 100,random_state=0)
regression.fit(level,salary)

print(regression.predict([[8.3]]))