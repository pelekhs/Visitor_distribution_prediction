#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 16:01:58 2018

function:
polygon filtering

@author: anastasis
"""
import pandas as pd
from shapely.geometry import Point
from defs import polygon1

my_df_ex = pd.read_csv('dataframe.csv', index_col = 0)
my_df_ex1 = my_df_ex

#CHECK THE POINTS (X,Y)
for i in range (0, len(my_df_ex1)):
    point = Point(my_df_ex.loc[i, 'X'],my_df_ex.loc[i,'Y'])
    if not polygon1.contains(point):
        my_df_ex1 = my_df_ex1.drop(my_df_ex.index[i])
'''
longtitude = X (8,....)
latitude = Y (48,....)
'''
#GET XY OF PEOPLE  
my_df_ex1 = my_df_ex1.drop_duplicates(subset='$oid', keep='first')
my_df_ex1 = my_df_ex1.sort_values(['Timestamp'])
my_df_ex1 = my_df_ex1.reset_index(drop=True)
my_df_ex1.to_csv('my_df_ex1.csv')