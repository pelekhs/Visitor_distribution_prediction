# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

#regression csv
csv = pd.read_csv('regression_error_15_15_SHIFT_1.csv')
results1 = (csv.iloc[:5,1:].values)
results=[]
for i in range(5):
    results.append([results1[i][x] for x in range(len(results1[i]))])
results=(np.asarray(results).astype(float))*100

#common
clusters= ['A','B','C','D','E','F']
cofs=[0,1,2,3]
ests = ['Random Forest','KNN','SVR','Kernel Ridge Regression','Deep Network']
cof=[]
cluster=[]
for cl in clusters:
    for i in cofs:
        cluster.append(cl)
        cof.append(str(i+1))

new = pd.DataFrame([cluster,cof]).transpose()
results=pd.concat([new,pd.DataFrame(results).transpose()],axis=1)

results.columns = ["Cluster","CoF"]+ests

#==============================================================================
# AVERAGE PLOTS
#==============================================================================
cof_mean=results.groupby(['CoF']).mean()
cof_mean1=cof_mean.copy()
cof_mean1["Average"]=cof_mean1.mean(axis=1)
cof_mean1.to_csv("Regressor_errors_per_CoF.csv")

cluster_mean=results.groupby(['Cluster']).mean()
cluster_mean1=cluster_mean.copy()
cluster_mean1["Average"]=cluster_mean1.mean(axis=1)
cluster_mean1.to_csv("Regressor_errors_per_Cluster.csv")


import matplotlib.pyplot as plt 
color = ['lightgreen', 'tomato', 'lightblue', 'goldenrod','grey'] 

for [mean,name] in list(zip([cof_mean,cluster_mean],["Combination of Features","Area of Interest"])):
    mean.plot(kind='bar', legend=True,width=0.4, figsize=(20,10),layout=[4,4], color=color)
    ax = plt.axes()
    plt.rcParams.update({'font.size': 20})
    plt.xticks(rotation='horizontal')
    plt.xlabel(name)
    plt.ylabel('Mean Relative Error (%)')
    ax.set_axisbelow(True)
    ax.grid(axis='y')
    ax.yaxis.grid(linestyle='-',linewidth=1.5,color='gray') # horizontal lines
    ax.xaxis.grid(b=False)
    plt.ylim(0,65)
    plt.legend(prop={'size': 16}, fancybox=True, framealpha=0.5)
    plt.savefig("Regressor_error_per_ "+name+".png")
    plt.show()

#==============================================================================
# #ACCUMULATED PLOT
#==============================================================================
data = ([list(results.iloc[:,x]) for x in range(2,7,1)])
label = results.columns[2:]
X = 1.3*(np.asarray([0,1,2,3,6,7,8,9,12,13,14,15,18,19,20,21,24,25,26,27,30,31,32,33])+1.7)
#X = np.asarray([0,1,2,3,10,11,12,13,20,21,22,23,30,31,32,33,40,41,42,43,50,51,52,53])
xticks=np.asarray([np.mean(X[i:i+4]) for i in range(0,len(X),4)])+0.6
minor_labels = np.repeat(['C1','C2','C3','C4'],len(clusters))
plt.figure()

for i in range(5):
#    if i%4==0:
    plt.bar(X+(i+1)*0.2, data[i], color = color[i], width = 0.2, label= label[i])
#    else:
#        plt.bar(X + i , data[i], color = color, width = 0.5)

ax = plt.axes()
plt.rcParams.update({'font.size': 20})
#plt.xticks(ticks=[1.5], labels=rotation='horizontal')
plt.xlabel('Area of Interest')
plt.ylabel('Mean Relative Error (%)')
ax.set_axisbelow(True)
ax.grid(axis='y')
ax.yaxis.grid(linestyle='-',linewidth=1.5,color='gray') # horizontal lines
ax.xaxis.grid(b=False)
#plt.xticks([1.5, 7.5, 13.5, 19.5, 25.5, 31.5], clusters ,rotation='horizontal')
plt.xticks(xticks, clusters ,rotation='horizontal')
plt.ylim(0,65)
plt.legend(prop={'size': 20},loc='upper right',fancybox=True, framealpha=0.5)
plt.show()
