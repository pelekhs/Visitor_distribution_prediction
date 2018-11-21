#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 15:21:13 2018

@author: anastasis
"""

import pandas as pd
from time_periods import periods

data = pd.read_csv('weather_history.csv')

temp = []
conditions = []

data["Time (GMT)"] = pd.to_datetime(data["Time (GMT)"])
data["Time (GMT)"] = data["Time (GMT)"].dt.strftime("%Y-%m-%d %H:%M:%S")

#Timestamp = line["Time (GMT)"]
#cond = line["Conditions"]
#temperature = line["Temp."]

def check_user(line):
    for i in range(0,len(data)):
        if (periods[line] >= data["Time (GMT)"].iloc[i] and periods[line] < data["Time (GMT)"].iloc[i+1]):
            temp.append(data["Temp."].iloc[i])
            conditions.append(data["Conditions"].iloc[i])
            break

for i in range(len(periods)-1):
    check_user(i)
    
L = ["Temperature", "Conditions"]

df = pd.DataFrame({'Temperature' : temp, 'Conditions' : conditions})