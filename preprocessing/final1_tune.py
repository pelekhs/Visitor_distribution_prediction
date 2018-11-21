# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 19:51:53 2018

@author: spele
"""
def final1_create_and_tune(data,pois_to_clusters,t11, t12, t21, t22,
                           artist_metrics,cluster,minutes,only_start='No',years='all'):
    from artists_period import create_artist
    import pandas as pd
    import numpy as np
    #define cluster distribution per period in the different clusters created
    
    final1 = cluster.cluster_distribution_per_period(data, min_users_per_period = 0)
    
    #create data structure of artist presence and popularity in clusters
    artists = create_artist(timestep=minutes, df=pois_to_clusters,only_start=only_start)
    
    if years != 'all':
        artists = artists[[(years in artists.index[k][0]) for k in range(len(artists))]]
        
    #I add the word pop for popularity next to the cluster name to identify the popularity
    #metric column of the artist that is performing at the corrrespondent cluster
    keys = list(artists.columns.values)
    fu = lambda x:x+"pop"
    if not(artist_metrics):#here i try to get rid of the metrics and just use the presence of the artists
        fu2 = lambda x:0 if x==0 else 1
        for key in keys:
            artists[key] = artists[key].apply(fu2) 
    values = list(map(fu,keys))
    dictionary = dict(zip(keys,values))
    artists = artists.rename(index=str, columns=dictionary)
    artists = artists.set_index(final1.index)
    #I concat to the training data
    frames = [final1,pd.DataFrame(artists,columns=list(artists.columns.values))]
    final1 = pd.concat(frames, axis = 1)
    
    final1[['Date', 'Period']] = final1['Period'].apply(pd.Series)
    
    #introducing extra col for time index and TS forecasting and year
    a = []
    year = []
    for index, row in final1.iterrows():
        a.append(row['Date'] +' '+row['Period'][:8])
        year.append(row['Date'][:4])

    final1['Time series dates']=a
    final1['Year'] =year    
    
    #######################          TIME FILTERS         ########################
    
    from defs import df_filtering
    filterer = df_filtering(final1)
    final1 = filterer.df_column_bounding(column_id='Period',lower_bound= t11+' to '+t12, upper_bound=t21+' to '+t22)
    if years!='all':
        final1 = filterer.df_column_bounding(column_id='Year',lower_bound=years, upper_bound=years)

#    #i use this temp dataframe to create some new lines in the begining of final1
#    #because first day starts from 5am not 12am so I fill with zeros at the beginning
#    if len(dates)>len(final1):
#        d = pd.DataFrame(np.zeros((len(dates)-len(final1), len(final1.columns))),columns=final1.columns)
#        final1=pd.concat([d,final1]).reset_index()
#        final1.drop("index",inplace=True,axis=1)
#    frames = [dates,final1]
#    final1 = pd.concat(frames, axis = 1)
    periods = final1.groupby(['Period']).size().index.tolist()
    period_indexes = list(range(len(periods)))
    dictionary3 = dict(zip(periods,period_indexes))
    final1['Time index'] = final1['Period'].map(lambda x:dictionary3[x])

    days = final1.groupby(['Date']).size().index.tolist()
    day_indexes = list(range(len(days)))
    day_indexes = list(x % 3 for x in day_indexes)
    dictionary2 = dict(zip(days,day_indexes))     
    final1['Day index'] = final1['Date'].map(lambda x:dictionary2[x])

#    final1.rename(columns={"index": "Time index"},inplace=True)
    #sort final1
    columns = list(final1.columns[1:].values)
    columns2= ['Date','Day index','Time index','Time series dates', 'Year']
    columns.remove('Date')
    columns.remove('Day index')
    columns.remove('Time index')
    columns.remove('Time series dates')
    columns.remove('Year')
    columns.sort()
    columns2.sort()
    cols=np.concatenate([columns,columns2])
    final1 = final1[cols]    
    #locate most significant cluster
    cltopred = str(final1.iloc[:,:-6].sum(axis=0).idxmax()).upper()
    #locate clusters with live shows
    clwithlive = list(dictionary.keys())
    print("\nClusters with live shows:", clwithlive)
    print("Most significant cluster = ", cltopred)
    
    return final1, cltopred, clwithlive