# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 22:05:13 2018

@author: spele
"""
#==============================================================================
# use jsonfences_an to filter points
#==============================================================================
#gaussian density plot
from scipy.stats import  gaussian_kde
from jsonFENCES_an import heat_lons, heat_lats, lons, lats
import numpy as np
from defs import mapread
import matplotlib.pyplot as plt


xy=np.vstack([heat_lons,heat_lats])
z = gaussian_kde(xy)(xy)
img = mapread()
x_border = lons
y_border = lats
fig, ax = plt.subplots()

if  ((len(x_border) != 0) and (len(y_border) != 0)):
    plt.xlim(min(x_border), max(x_border))
    plt.ylim(min(y_border), max(y_border))

    if (img != ''):
        plt.imshow(img, alpha=0.5, extent=[x_border[0],x_border[2],y_border[0], y_border[2]])
else:
    if (img != ''):
        plt.imshow(img, alpha=0.5)

from matplotlib.ticker import FormatStrFormatter
plt.scatter(heat_lons, heat_lats, alpha=0.9, cmap='jet', s=1, c=z, edgecolor='')
plt.ylabel('Latitude')
plt.xlabel('Longitude')
ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
plt.show()
plt.savefig('gaussian_density_plot_2017_2018.png')