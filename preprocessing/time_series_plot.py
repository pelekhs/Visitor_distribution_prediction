# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 19:16:58 2018

@author: spele
"""

#==============================================================================
# time series plot
#==============================================================================
#live and popularities plot in time
import matplotlib.pyplot as plt
import matplotlib.dates as md
import pandas as pd
from datetime import datetime

#insert final_multi made by papework2.py of your choice to continue
#if multiplier==1 then final_multi=final2
#from papework2 import minutes, multiplier, final_multi

#examples of variables needed by the script so as not to import from papework2.py
final_multi = pd.read_csv('final_multi_5to30mins.csv')
minutes = 5
multiplier = 6 

index=[x for x in final_multi['Time series']]
index2 = [x for x in final_multi['Time series']]

#==============================================================================
# select clusters to plot and assign column to the corresponding varable
#==============================================================================
tuTS = pd.Series(final_multi['Total users'].values,index=index)
#conditions variable
condTS = pd.Series(final_multi['Conditions'].values, index=index2)
#temperature variable
tempTS = pd.Series(final_multi['Temperature'].values, index=index2)
#main cluster with live (center)
liveTS = pd.Series(final_multi['A'].values,index=index)
#popularity of cluster with live
popTS = pd.Series(final_multi['Apop'].values,index=index)
#cluster that contains entrance
exitTS = pd.Series(final_multi['E'].values,index=index)
#shoopping cluster
shopTS = pd.Series(final_multi['F'].values,index=index)

##normalize
#maxpeople=max(tuTS)
#tuTS = tuTS/(maxpeople)
#liveTS = liveTS/(maxpeople)
#exitTS = exitTS/(maxpeople)
#shopTS = shopTS/(maxpeople)

pt = [(datetime.strptime(tuTS.index[x], '%Y-%m-%d %H:%M:%S')) for x in range(len(tuTS))]
pt2 = [(datetime.strptime(tuTS.index[x], '%Y-%m-%d %H:%M:%S')) for x in range(len(condTS))]
     
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
#condsc = StandardScaler()
#condTS = condsc.fit_transform(condTS.values.reshape(-1,1))
#tempsc = StandardScaler()
#tempTS = tempsc.fit_transform(tempTS.values.reshape(-1,1))
#weatherTS = tempTS+condTS
#weathersc = MinMaxScaler()
#weatherTS = tempsc.fit_transform(weatherTS.values.reshape(-1,1))
#weatherTS = pd.Series(np.repeat(weatherTS,multiplier), index=index)

condsc = MinMaxScaler()
condTS = condsc.fit_transform(condTS.values.reshape(-1,1))

fig, ax1 = plt.subplots()
ax1.set_xlabel('Time',fontsize=22)
ax1.set_ylabel('Number of people among AoIs',fontsize=22)


ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#ax2 = ax1

ax1.plot(pt, tuTS, color = 'black', label ='Total Users', linewidth='2.5', dashes=[15,5,5,15,5,5])
ax1.plot(pt, liveTS, label = 'Main stage',linewidth='2.5',color = 'blue', dashes=[6,3,6,6,3,6])
ax1.plot(pt, exitTS, label = 'Entrance/Exit',linewidth='2.5',color = 'red', linestyle='-')
ax1.plot(pt, shopTS, label = 'Shopping',linewidth='2.5',color = 'green', dashes=[4,2,4,4,2,4])
ax1.set_ylim(bottom=0, top=max(tuTS)+0.05*max(tuTS))
ax2.set_ylim(bottom=0, top=max(condTS)+0.1*max(condTS))

ax2.set_ylabel('', color='black',fontsize=22)
ax2.plot(pt, condTS, color='cyan',label = 'Conditions', linestyle = '--', linewidth='1.5')       
ax2.plot(pt, popTS, color='black',label = 'Popularity', linestyle = '-.', linewidth='2.3')  
ax2.set_ylabel('Normalized Weather and Popularity',fontsize=22)


## Set time format and the interval of ticks
xformatter = md.DateFormatter('%m-%d %H:%M')
xlocator = md.MinuteLocator(interval = 180)
minorxlocator = md.MinuteLocator(interval = int(minutes*multiplier))

## Set xtick labels to appear every 15 minutes
ax1.xaxis.set_major_locator(xlocator)
ax1.xaxis.set_minor_locator(minorxlocator)
for x in ax1.xaxis.get_major_ticks():
    x.label.set_fontsize(20)
for y in ax1.yaxis.get_major_ticks():
    y.label.set_fontsize(20)

## Format xtick labels as HH:MM
plt.gcf().axes[0].xaxis.set_major_formatter(xformatter)
# Customize the major grid
ax1.grid(which='major', linestyle=':', linewidth='0.6',color='gray')
# Customize the minor grid
ax1.grid(which='minor', linestyle='-', linewidth='0.15', color='gray')

plt.setp(ax2.get_yticklabels(),fontsize=20)
fig.autofmt_xdate()
ax1.legend(fontsize=22, prop={'size': 22},loc='upper left')
ax2.legend(loc=1,fontsize=22, prop={'size': 22})
plt.show()