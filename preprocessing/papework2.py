
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 22:40:23 2018

@author: spele
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 16:04:51 2018

@author: spele
"""

import numpy as np
import pandas as pd
#==============================================================================
# ###########     choose parameters that affect the primary dataset     #######
#==============================================================================
#number of clusters and time period interval and filtering hours
clusters = 6
t11, t12, t21, t22 = '00:00:00','00:15:00','23:45:00','00:00:00'

#choose between 'all '2017' '2018'
years = '2017'


from datetime import datetime
t1 = datetime.strptime(t11, '%H:%M:%S')
t2 = datetime.strptime(t12, '%H:%M:%S')
minutes = t2.minute-t1.minute

#create datasets
from defs_after import clustersPeriods
cluster = clustersPeriods(clusters=clusters, timestep=minutes)
data, pois_to_clusters = cluster.assign_users_in_regions(df2 = "artists_list.csv",
                                                         export_csv = False,
                                                         plot = False, years = years,                                         
                                                         random_state = 42, clustering = 'kmeans') #new dataset in dataframe_teliko.csv
#==============================================================================
# #############    Parameters below affect creation of final1     #############
#==============================================================================
#False if you only want his presence
artist_metrics = True
# 'No':metric during live, 'start':metric only at start, 'end':metric only before the bands live
only_start='No'

from final1_tune import final1_create_and_tune
final1, cltopred, clwithlive = final1_create_and_tune(data=data,
                                            pois_to_clusters=pois_to_clusters,
                                            t11=t11, t12=t12, t21=t21, t22=t22,
                                            artist_metrics=artist_metrics,
                                            cluster=cluster,minutes=minutes,
                                            only_start=only_start,years=years)
if (cltopred not in clwithlive):
   cltopred = str(input('NO live in most significant cluster. Enter cluster to predict:\n')).upper()
print("I predict: ", cltopred)


# =============================================================================
# CREATE WEATHER COLUMNS
# =============================================================================
periods = final1["Time series dates"].tolist()
data = pd.read_csv('weather history.csv')
temp = []
conditions = []
data["Time (GMT)"] = pd.to_datetime(data["Time (GMT)"])
data["Time (GMT)"] = data["Time (GMT)"].dt.strftime("%Y-%m-%d %H:%M:%S")

def check_user(line):
    for i in range(0,len(data)):
        if (periods[line] >= data["Time (GMT)"].iloc[i] and periods[line] < data["Time (GMT)"].iloc[i+1]):
            temp.append(data["Temp."].iloc[i])
            conditions.append(data["Conditions"].iloc[i])
            break

for j in range(len(periods)):
    check_user(j)
    
L = ["Temperature", "Conditions"]

df = pd.DataFrame({'Temperature' : temp, 'Conditions' : conditions, 'Periods' : periods})
df.set_index('Periods', inplace = True)

#    cond = list(df['Conditions'].unique())
#    cond_dict = {el:0 for el in cond}
import yaml
#    with open('cond_dict.yml', 'w') as outfile:
#        yaml.dump(cond_dict, outfile, default_flow_style=False)
with open('cond_dict.yml', 'r') as stream:
    cond_dict = yaml.load(stream)



df.reset_index(drop=True, inplace=True)
df['Temperature'] = df['Temperature'].apply(lambda x: int(x[:2]))
df['Conditions'] = df['Conditions'].apply(lambda x: cond_dict[x])


#==============================================================================
# ######################     Dataset to be scaled       #######################
#==============================================================================
#final2 will be the normalized version of final1 

clusters= ['A','B','C','D','E','F']
final2 = pd.DataFrame()
final2[clusters]=final1[clusters]
for cl in clwithlive:
    final2[cl+"pop"]=final1[cl+"pop"]
final2['Total users'] = final1['Total users']
final2['Temperature'] = df['Temperature']
final2['Conditions'] = df['Conditions']
if years=='all':
    final2['Year'] = final1['Year']
final2['Time index'] = final1['Time index']

#final2['Time series']=final1['Time series dates']

###optional if you want time series dates
final2['Time series']=final1['Time series dates']
#%%#
#==============================================================================
# #FROM 5 MINUTES TO 30 MINUTES
#==============================================================================
from theregulator import theregulator
multiplier=1
final_multi = theregulator(final2, minutes, multiplier,clwithlive)

#order columns
columns=np.concatenate([clusters,[clwithlive[0]+'pop',clwithlive[1]+'pop','Total users', 
                                  'Temperature', 'Conditions', 'Time index', 'Time series']])
final_multi = final_multi[columns]
final_multi=final_multi.append(final_multi.iloc[-1][:]).reset_index(drop=True)
#final_multi.iloc[-1,-1] = '2018-07-23 00:00:00'
final_multi.iloc[-1,-1] = '2017-07-24 00:00:00'
#final2=final2[columns[:-1]]
#==============================================================================
# FROM VALUES IN CLUSTERS TO DISTRIBUTIONS
#==============================================================================
##distribution normalization
#for column in clusters:
#    final2[column] = final2[column]/final2['Total users']
#final2 = final2.fillna(0)

#==============================================================================
# ########################    Feature Scaling    ##############################
#==============================================================================
#from sklearn.preprocessing import MinMaxScaler
#mini=0
#maxi=1

##total users feature scaling
#sc_tot = MinMaxScaler(feature_range=(mini, maxi))
#final2["Total users"] = sc_tot.fit_transform(final2["Total users"].values.reshape(-1,1))
#
##time index as a sine shifted so that it contains only positive values
#final2['Periodic time index'] = final2["Time index"].apply(lambda x: (np.sin((2*np.pi*x/(max(final2["Time index"]))))))
#
##time index feature scaling    
#sc_index = MinMaxScaler(feature_range=(mini, maxi))
#final2["Time index"] = sc_index.fit_transform(final2["Time index"].values.reshape(-1,1))

##year encoding
#if years=='all':
#    from sklearn.preprocessing import LabelEncoder
#    le = LabelEncoder()
#    final2['Year']=le.fit_transform(final2['Year'])
#final2 = (final2.drop(labels='Year',inplace=True,axis=1)).reset_index(drop=True)
##popularity feature scaling
#from artists_period import create_artist
#artists = create_artist(timestep=minutes, df=pois_to_clusters, only_start=only_start)
#sc_pop = MinMaxScaler(feature_range=(mini, maxi))
#sc_pop.fit(pd.concat([artists.iloc[:,0], artists.iloc[:,1]]).values.reshape(-1, 1))
#for cltopred in clwithlive:
#    final2[cltopred+"pop"] = sc_pop.transform(final2[cltopred+"pop"].values.reshape(-1,1))
#
##temperature feature scaling
#sc_temp = MinMaxScaler(feature_range=(mini, maxi))
#final2["Temperature"] = sc_tot.fit_transform(final2["Temperature"].values.reshape(-1,1))
#
##time index feature scaling    
#sc_index = MinMaxScaler(feature_range=(mini, maxi))
#final2["Conditions"] = sc_index.fit_transform(final2["Conditions"].values.reshape(-1,1))

#==============================================================================
# time series plot
#==============================================================================
#%%#
#live and popularities plot in time
import matplotlib.pyplot as plt
import matplotlib.dates as md

index=[x for x in final_multi['Time series']]
index2 = [x for x in final_multi['Time series']]

tuTS = pd.Series(final_multi['Total users'].values,index=index)
condTS = pd.Series(final_multi['Conditions'].values, index=index2)
tempTS = pd.Series(final_multi['Temperature'].values, index=index2)
liveTS = pd.Series(final_multi['B'].values,index=index)
popTS = pd.Series(final_multi['Bpop'].values,index=index)
exitTS = pd.Series(final_multi['A'].values,index=index)
shopTS = pd.Series(final_multi['E'].values,index=index)

#normalize
maxpeople=max(liveTS)
tuTS = tuTS/(maxpeople)
liveTS = liveTS/(maxpeople)
exitTS = exitTS/(maxpeople)
shopTS = shopTS/(maxpeople)

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
ax1.set_ylabel('Normalized Values',fontsize=22)


#ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2 = ax1

#ax1.plot(pt, tuTS, color = 'black', label ='Total Users', linewidth='2.5', dashes=[15,5,5,15,5,5])
ax1.plot(pt, liveTS, label = 'Main stage',linewidth='2.5',color = 'blue', dashes=[6,3,6,6,3,6])
ax1.plot(pt, exitTS, label = 'Entrance/Exit',linewidth='2.5',color = 'red', linestyle='-')
#ax1.plot(pt, shopTS, label = 'Shopping',linewidth='2.5',color = 'green', dashes=[4,2,4,4,2,4])
ax1.set_ylim(bottom=0, top=max(liveTS)+0.05*max(liveTS))
#ax2.set_ylim(bottom=0, top=max(condTS)+0.1*max(condTS))

#ax2.set_ylabel('', color='black',fontsize=22)
#ax2.plot(pt, condTS, color='cyan',label = 'Conditions', linestyle = '--', linewidth='1.5')       
ax2.plot(pt, popTS, color='black',label = 'Popularity', linestyle = '-.', linewidth='2.3')  
#ax2.set_ylabel('Normalized Weather Conditions',fontsize=22)


## Set time format and the interval of ticks (every 15 minutes)
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
#datemin = datetime(2017,7,21,0,0)
#datemax = datetime(2017,7,24,0,0)
#ax1.set_xlim(datemin, datemax)
#plt.xticks(rotation=45)
fig.autofmt_xdate()
ax1.legend(loc=0,fontsize=22, prop={'size': 22})
#ax2.legend(loc=0,prop={'size': 23})
plt.show()
#%%#
total_users_friday = sum(final1['Total users'][final1['Day index']==0])
print(total_users_friday)
total_users_saturday = sum(final1['Total users'][final1['Day index']==1])
print(total_users_saturday)
total_users_sunday = sum(final1['Total users'][final1['Day index']==2])
print(total_users_sunday)
# =============================================================================
# x, y WRITE TO CSV
# =============================================================================
x = final_multi.iloc[:-1,:].reset_index(drop=True)
y = final_multi[clusters].iloc[1:,:].reset_index(drop=True)

x.to_csv('x_'+years+'.csv', index=False)
y.to_csv('y_'+years+'.csv', index=False)


#==============================================================================
# IMPORT WRITTEN CSV
#==============================================================================
x_2018 = pd.read_csv('x_2018.csv')
y_2018 = pd.read_csv('y_2018.csv')

x_2017 = pd.read_csv('x_2017.csv')
y_2017 = pd.read_csv('y_2017.csv')

#==============================================================================
# PREDICTIONS
#==============================================================================
for cluster in clusters:
    X_train = x_2017
    y_train = y_2017[cluster]
    
#==============================================================================
# #############      Model Creation, Evaluation and Plots     #################
#==============================================================================
#regression model parameters to define
#SVR 
svr_kernel_vector= ['rbf','linear']#poly rbf linear sigmoid precomputed
#random forest numbers of estimators to use for regression
estimators = [10]
#polynomial regression maximum degree
poly_deg = [1]

from models_evaluation_and_plots import model_evaluation_and_plots
modeler = model_evaluation_and_plots(X_train_sc_flat,X_test_sc_flat,y_train_sc, ytrue, y_train, y_test,
            sc_cl)
#SVR
svr=[]
for svr_kernel in svr_kernel_vector:
    svr.append(modeler.SVR(svr_kernel))
#Random Forest
rf=[]
for estimator in estimators:
    rf.append(modeler.RF(estimator,max_depth=2, min_samples_split=4))
#polynomial
po=[]
for deg in poly_deg:
    po.append(modeler.poly(deg))
#MLP
mlp=[modeler.MLP()]
#svr, rf, po, mlp are formated as below:
#[yhat, ytrain_hat, ytest_hat, Sete, Setr,Se, R2te, R2tr, R2, mapete, mapetr, mape]


#anastasis
#anasdataset
Xa, Xb, y, ya, yb, ya_sc, yb_sc =\
sets2(1,final1['Time index'],output_cols,output_cols_sc,dataset,3)
X_sc = np.concatenate([Xa, Xb])
y_sc = np.concatenate([ya_sc, yb_sc])

#shift (not a model just a shift)
shift=[modeler.shift(1,output_cols,final1['Time index'],testing_day,years=years)]
#shift is formatted as below
#yhat, ytrain_hat, ytest_hat, Sete, Setr,Se, R2te, R2tr, R2, mapete, mapetr, mape
yhatshift=shift[0][0]
ytrueshift=shift[0][1]
