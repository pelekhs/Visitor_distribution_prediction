# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 16:36:11 2017

This code produces the variables 
heat_lats, heat_lons: Y,X of people in the fest
poi_lons, poi_lats:   Y,X of POIs
polygon1, lats, lons: Polygon of the area of the fest, Y,X of its vertices(akmes)

There can be a selection of heat_lats, heat_lons poi_lons, poi_lats in case we apply
the filters that are grey comments to my_df_ex1 and poi pandas dataframes. 

At the end it draws a voronoi plot of the selected PoIs and stores it into voronoi_plot.png image file

@author: ANASTASIS
"""
import pandas as pd

#df_ex_1 contains space and time filtered data with duplicates
my_df_ex1 = pd.read_csv('my_df_ex1.csv')
#katargo duplicates me idia wra kai idio user
my_df_ex1_withoutduplicates = my_df_ex1.drop_duplicates(subset=["ID","Epoch"], keep='first')
my_df_ex1_withoutduplicates.reset_index(inplace=True)

poi = pd.read_json('festinfrastructure.json', lines='true')

"""uncomment/comment below to activate/deactivate the dataframe filters""" 
"""from here:"""
#optional import dataframe filtering class from defs.py file & create object
from defs import df_filtering
filterer = df_filtering(my_df_ex1_withoutduplicates)
#optional filtering of column of your choice with lower and upperbound(mainly for use on users dataframe)
my_df_ex2018 = filterer.df_column_bounding(column_id='Date',lower_bound= "2018-07-20", upper_bound="2018-07-22")
my_df_ex2017 = filterer.df_column_bounding(column_id='Date',lower_bound= "2017-07-21", upper_bound="2017-07-23")
##optional filter2:(mainly for use on POI dataframe)
##keep only rows that contain a particular string in a specific column and dict_key 
##arguments(df to filter, column to look in(column contains dictionary),dictionary key, string to look for )"""
#filterer2 = df_filtering(poi)
#poi = filterer2.column_check_str('properties', 'layer', 'BΓΌhnen & Areale')
##for Bühnen & Areale type 'BΓΌhnen & Areale'
"""till here"""

from defs import getpoi, getxy, polygonaki
#Y,X vectors for all active users in the fest
heat_lons, heat_lats = getxy(my_df_ex1_withoutduplicates)
heat_lons2018, heat_lats2018 = getxy(my_df_ex2018)
heat_lons2017, heat_lats2017 = getxy(my_df_ex2017)
#Y,X vectors for all POIs inside the fest area
poi_lons, poi_lats = getpoi(poi)     

#polygon of the border, coordinates of its vertices
polygon1, lons, lats = polygonaki()

#==============================================================================
# PLOTS BELOW
#==============================================================================

#==============================================================================
#     2017,2018 together
#==============================================================================
if __name__ == "__main__":
    #gaussian density plot
    from scipy.stats import  gaussian_kde
    import numpy as np
    from defs import mapread
    import matplotlib.pyplot as plt
    
    plt.figure("2017-2018")
    xy=np.vstack([heat_lons,heat_lats])
    z = gaussian_kde(xy)(xy)
    img = mapread()
    x_border = lons
    y_border = lats
    
    if  ((len(x_border) != 0) and (len(y_border) != 0)):
        plt.xlim(min(x_border), max(x_border))
        plt.ylim(min(y_border), max(y_border))
    
        if (img != ''):
            plt.imshow(img, alpha=0.5, extent=[x_border[0],x_border[2],y_border[0], y_border[2]])
    else:
        if (img != ''):
            plt.imshow(img, alpha=0.5)
            
    plt.scatter(heat_lons, heat_lats, alpha=0.9, cmap='jet', s=1, c=z, edgecolor='')
    plt.show()
    plt.savefig('gaussian_density_plot_2017_18.png')
#==============================================================================
#     2018
#==============================================================================
    heat_lons, heat_lats = heat_lons2018, heat_lats2018
        #gaussian density plot

    plt.figure("2018")
    xy=np.vstack([heat_lons,heat_lats])
    z = gaussian_kde(xy)(xy)
    img = mapread()
    x_border = lons
    y_border = lats
    
    if  ((len(x_border) != 0) and (len(y_border) != 0)):
        plt.xlim(min(x_border), max(x_border))
        plt.ylim(min(y_border), max(y_border))
    
        if (img != ''):
            plt.imshow(img, alpha=0.5, extent=[x_border[0],x_border[2],y_border[0], y_border[2]])
    else:
        if (img != ''):
            plt.imshow(img, alpha=0.5)
            
    plt.scatter(heat_lons, heat_lats, alpha=0.9, cmap='jet', s=1, c=z, edgecolor='')
    plt.show()
    plt.savefig('gaussian_density_plot_2018.png')
#==============================================================================
#     2017 only
#==============================================================================
    heat_lons, heat_lats = heat_lons2017, heat_lats2017

    plt.figure("2017")
    xy=np.vstack([heat_lons,heat_lats])
    z = gaussian_kde(xy)(xy)
    img = mapread()
    x_border = lons
    y_border = lats
    
    if  ((len(x_border) != 0) and (len(y_border) != 0)):
        plt.xlim(min(x_border), max(x_border))
        plt.ylim(min(y_border), max(y_border))
    
        if (img != ''):
            plt.imshow(img, alpha=0.5, extent=[x_border[0],x_border[2],y_border[0], y_border[2]])
    else:
        if (img != ''):
            plt.imshow(img, alpha=0.5)
            
    plt.scatter(heat_lons, heat_lats, alpha=0.9, cmap='jet', s=1, c=z, edgecolor='')
    plt.show()
    plt.savefig('gaussian_density_plot_2017.png')
    
    
    #DIY voronoi plot
    
    #import numpy as np
    #import matplotlib.pyplot as plt
    #from scipy.spatial import Voronoi
    #from voronoi import voronoi_finite_polygons_2d
    #from defs import mapread
    
    
    #plt.figure()
    #img = mapread()
    #x_border = lons
    #y_border = lats
    #
    #x = np.column_stack([poi_lons,poi_lats])
    #vor = Voronoi(x)
    #regions, vertices = voronoi_finite_polygons_2d(vor)
    #color=['orange', 'green', 'blue', 'purple', 'red', 'yellow', 'magenta', 'black', 'cyan', 'grey', 'pink', 'beige']
    
    ##dimension fix and map plot
    #i = 0
    #for region in regions:
    #    we = vertices[region]
    #    plt.fill(*zip(*we), alpha=0.2, c=color[i % 11], edgecolor='black', linewidth = 1)
    #    i += 1
    #
    #if  ((len(x_border) != 0) and (len(y_border) != 0)):
    #    plt.xlim(min(x_border), max(x_border))
    #    plt.ylim(min(y_border), max(y_border))
    #
    #    if (img != ''):
    #        plt.imshow(img, alpha=0.5, extent=[x_border[0],x_border[2],y_border[0], y_border[2]])
    #else:
    #    if (img != ''):
    #        plt.imshow(img, alpha=0.5)
    #plt.scatter(poi_lons, poi_lats, edgecolor = '', s = 15, c = 'black')        
    #plt.show()
    #plt.savefig('voronoi_plot.png')
