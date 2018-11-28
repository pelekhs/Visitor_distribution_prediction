# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 22:02:53 2018

@author: spele
"""

#DIY voronoi plot
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from voronoi import voronoi_finite_polygons_2d
from clustering_belos import my_kmeans
from defs import mapread

#here we import 2 np arrays of longitudes and latitudes of the points that we want to plot the voronoi diagram 
from jsonFENCES_an import lons, lats, poi_lons, poi_lats

plt.figure()
img = mapread()
x_border = lons
y_border = lats

x = np.column_stack([poi_lons,poi_lats])
vor = Voronoi(x)
regions, vertices = voronoi_finite_polygons_2d(vor)
#vector of colors to be used
color=['orange', 'green', 'blue', 'purple', 'red', 'yellow', 'magenta', 'black', 'cyan', 'grey', 'pink', 'beige']

#dimension fix and map plot
i = 0
for region in regions:
    we = vertices[region]
    plt.fill(*zip(*we), alpha=0.2, c=color[i % 11], edgecolor='black', linewidth = 1)
    i += 1

if  ((len(x_border) != 0) and (len(y_border) != 0)):
    plt.xlim(min(x_border), max(x_border))
    plt.ylim(min(y_border), max(y_border))

    if (img != ''):
        plt.imshow(img, alpha=0.5, extent=[x_border[0],x_border[2],y_border[0], y_border[2]])
else:
    if (img != ''):
        plt.imshow(img, alpha=0.5)
plt.scatter(poi_lons, poi_lats, edgecolor = '', s = 15, c = 'black')        
plt.show()
#plt.savefig('voronoi_plot.png')