#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 15:09:14 2018

@author: anastasis
"""

import numpy as np
import pandas as pd
import json
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
import os
cwd = os.getcwd()

class df_filtering:
    def __init__(self, df):
        if ".csv" in df:
            self.df = pd.read_csv(df).iloc[:,1:]# because CSVs have also stored indexes as an extra column at the begining
        else:
            self.df = df
#df:dataframe to be filtered, column_id: name or index of column to be filtered, lower & upper bound: min and max bounds between which elements are kept. All others are dropped
    def df_column_bounding(self, column_id, lower_bound, upper_bound):
        if ".csv" in self.df:
            data = pd.read_csv(self.df,index_col=0)# because CSVs have also stored indexes as an extra column at the begining
        else:
            data = self.df
        data = data[(data.astype(str)[column_id] >= lower_bound) & (data.astype(str)[column_id] <= upper_bound)]
        data = data.reset_index(drop=True)
        return data
        
#checks for string in dataframe of the festinfrastructure.json format and drops others    
    def column_check_str(self, column_id, dict_key, string_to_check_for):
        temp = self.df
        for i in range (0,len(self.df)):
            if (self.df[column_id][i][dict_key] != string_to_check_for):
                temp = temp.drop(self.df.index[i])
        temp = temp.reset_index(drop=True)
        return temp
        
#returns the polygon in which the fest happens(polygon1) as well as the 
#coordinates of its vertices Y,X       
def polygonaki():
    jsondata = json.load(open('Geofences.json'))
    jsondata1 = jsondata['features'][0]['geometry']['coordinates']

    p1 = jsondata1[0][0]
    p2 = jsondata1[0][1]
    p3 = jsondata1[0][2]
    p4 = jsondata1[0][3]
    p5 = jsondata1[0][4]

    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    p3 = np.asarray(p3)
    p4 = np.asarray(p4)
    p5 = np.asarray(p5)

#    polygon = np.array([p1,p2,p3,p4,p5])

    lons = np.array([p1[0],p2[0],p3[0],p4[0],p5[0]])
    lats = np.array([p1[1],p2[1],p3[1],p4[1],p5[1]])

    lons_lats_vect = np.column_stack((lons,lats))

    polygon1 = Polygon(lons_lats_vect)
    
    return polygon1, lons, lats

polygon1, lons, lats = polygonaki()

def mapread():
    from imageio import imread
    import matplotlib.cbook as cbook
    datafile = cbook.get_sample_data(cwd+'//prosanatolismeno_dasfest map.jpg')
    img = imread(datafile)
    return img
    
#this function returns two independent numpy arrays with the 2 columns of a 
#dataframe with column names 'Y' and 'X' in this order (Y,X)     
def getxy(my_df_ex1):
    heat_lats = []
    heat_lons = []
    for i in range (0, len(my_df_ex1)):
        heat_lats.append(my_df_ex1.loc[i]['Y'])
        heat_lons.append(my_df_ex1.loc[i]['X'])
    heat_lats = np.array(heat_lats)
    heat_lons = np.array(heat_lons)
    return heat_lons,heat_lats
    
    
#same as getxy but for dataframe formatted as Basmati with the Pois
def getpoi(poi_df):        
    #check for the coordinates of POI
    poix=[]
    poiy=[]
    for i in range(len(poi_df)):
        poix.append(poi_df.iloc[i][1]['coordinates'][0])
        poiy.append(poi_df.iloc[i][1]['coordinates'][1])

    poi_lons = np.array(poix)
    poi_lats = np.array(poiy)
    lons_lats_vect_poi = np.column_stack((poi_lons,poi_lats))

    poix1 = []
    poiy1 = []

    for i in range (0, len(lons_lats_vect_poi)):
        point = Point(lons_lats_vect_poi[i][0], lons_lats_vect_poi[i][1])
        if polygon1.contains(point):
            poix1.append(lons_lats_vect_poi[i][0])
            poiy1.append(lons_lats_vect_poi[i][1])
        
        
        poi_lons = np.array(poix1)
        poi_lats = np.array(poiy1)
    return poi_lons, poi_lats
    
#poi_lons, poi_lats = getpoi(poi2)