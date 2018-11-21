#H sinartisi ftiaxnei katanomi xristwn sta opoia clusters se kathe xroniki periodo
#dexetai to dataframe me toulaxiston 2 stiles pou na periexoun periods kai clusters
#mporoume na orisoume kai poies einai aftes oi stiles meso twn col1 col2
#episis eisagoume an theloume k ena katotato orio users per period wste na lifthei afti i periodos
#ip opsi

    
class clustersPeriods():

    def __init__(self, clusters = 6, timestep = 15):
        self.clusters = clusters
        self.timestep = timestep
    #df can be .csv or pandas dataframe    
    def cluster_distribution_per_period(self, df = 'dataframe_teliko.csv',
                                        min_users_per_period = -1):
        import pandas as pd
        timestep = self.timestep
        if ".csv" in df:
            dataset = pd.read_csv(df).iloc[:,1:]# because CSVs have also stored indexes as an extra column at the begining
        else:
            dataset = df
        dataset = dataset[['Date', 'Period', 'Cluster']]
        active_periods = dataset.groupby(['Date','Period']).size()
        
    #edw pairnw to df pou periexei oles tis xronikes periodous kai to fillarw me tis times tou
    #active_periods, etsi wste na exw pleon ta panta!!!
        from time_periods import create
        periods_from_to = create(timestep)
        a = dataset.groupby(['Date']).size().index.tolist()
        import itertools
        periods_from_to = list((itertools.product(a, periods_from_to)))
#        periods_from_to = active_periods.index.values.tolist()
        df = pd.Series(index=periods_from_to)
        for i in range(len(df)):
            try:
                df[i] = active_periods.get(df.index[i])
            except KeyError:
                df[i] = 0
        df = df.fillna(0)
    #twra vazw to active periods = df gia na exw ola ta active periods
        active_periods = df
        active_periods = active_periods[active_periods[:] >= min_users_per_period]
        users_per_period = list(active_periods)
        active_periods = active_periods.index.tolist()
        dataset['Period'] = list(zip(dataset.Date, dataset.Period))
        dataset = dataset[dataset['Period'].isin(active_periods)]
        dataset = dataset.reset_index(drop = True)
        
        b = list(dataset.groupby('Cluster').groups.keys())
        
        columns = ['Period']
        clusters = {}
        for letters in b:
            clusters.update({letters: []})
            columns.append(letters)
            
        for period in active_periods:
            a = dataset[dataset['Period'] == period].groupby('Cluster').size()
            for cluster in clusters:
                try:
                    clusters[cluster].append(a[cluster])
                except:
                    IndexError
                    clusters[cluster].append(0)
        
        final = pd.DataFrame(columns=columns)
        
        final['Period'] = active_periods
        for cluster in clusters:
            final[cluster] = clusters[cluster]
        final['Total users'] = users_per_period
        return(final)
        
        
    #assign each user in the chosen number of clusters (cluster_numbers). Create clusters
    #based on alphabet letters (A, B, C, ...). Save dataset into csv. Plot the map to see 
    #what happens in case you want to. 
    
    def assign_users_in_regions(self, df1 = "dataframe_1.csv", df2 = None, plot = False, export_csv = False, years = '2017', clustering='kmeans', random_state = None):
        import pandas as pd
        from shapely.geometry.polygon import Polygon
        from voronoi import voronoi_finite_polygons_2d
        import matplotlib.pyplot as plt
        from scipy.spatial import Voronoi
        from shapely.geometry import Point
        from clustering_belos import my_kmeans
        from jsonFENCES_an import heat_lons, heat_lats, heat_lons2018, heat_lats2018, heat_lons2017, heat_lats2017, lons, lats
        from defs import mapread
        from time_periods import create
        timestep = self.timestep
        create(timestep)
        if years =='2017':
            heat_lons = heat_lons2017
            heat_lats = heat_lats2017 
        elif years =='2018':
            heat_lons = heat_lons2018
            heat_lats = heat_lats2018
        
        if clustering == 'kmeans':
            #kmeans clustering
            kmeansObject = my_kmeans(heat_lons, heat_lats)
            kmeansObject.clustering(clusters = self.clusters, x_border = lons,
                                    y_border = lats, random_state = random_state,
                                    plot = False)
            centers = kmeansObject.centers
            
        if clustering == 'meanshift':
            #meanshift
            #class sklearn.cluster.MeanShift(bandwidth=None, seeds=None, bin_seeding=False, min_bin_freq=1, 
                                             #cluster_all=True, n_jobs=None
            from sklearn.cluster import MeanShift
            import numpy as np
            y = range(len(heat_lons))
            X = np.column_stack((heat_lons, heat_lats,y))
            mean_shifter = MeanShift(bandwidth=0.00098, bin_seeding=False, min_bin_freq=1, cluster_all=True, n_jobs=-1)
            X[:,2] = mean_shifter.fit_predict(X[:,:2])
            centers = mean_shifter.cluster_centers_
            
        #index_col = 0 gia na min emfanizei tin extra stili Unnamed..
        dataset = pd.read_csv(df1, index_col = 0)
        from defs import df_filtering
        filterer = df_filtering(dataset)
        if years == '2017':
            dataset = filterer.df_column_bounding(column_id='Date',lower_bound= "2017-07-21", upper_bound="2017-07-23")
        if years == '2018':
            dataset = filterer.df_column_bounding(column_id='Date',lower_bound= "2018-07-20", upper_bound="2018-07-22")
        
        vor=Voronoi(centers)
        regions, vertices = voronoi_finite_polygons_2d(vor)
    
        polygon = []
        for region in regions:
            we = vertices[region]
            polygon.append(Polygon(we))
    
        tetragwno = []
        def check_polygon(X, Y):
            point = Point(X, Y)
            for i in range(len(centers)):
                if polygon[i].contains(point):
                    tetragwno.append(chr(i+65))
                    break
            
        for i in range(len(dataset)):
            check_polygon(dataset['X'].iloc[i], dataset['Y'].iloc[i])
        dataset['Cluster'] = tetragwno
        if export_csv:
            dataset.to_csv('dataframe_teliko.csv')
        
        tetragwno = []
        if (df2 != None):
            pois = pd.read_csv(df2, index_col = 0)#these are the pois that I want to assign to the clusters created from the above dataset. I dont want them to influence the clustering process
            pois2 = pd.read_csv(df2)
            for i in range(len(pois)):
                check_polygon(pois['X'].iloc[i], pois['Y'].iloc[i])
            pois['Cluster'], pois2['Cluster'] = tetragwno, tetragwno
            pois_to_clusters = pois2            
            if export_csv:
                pois.to_csv('pois_toclusters.csv')    
            
        plt.figure()
        img = mapread()
        z=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
           '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', 'black']
           
        for region in regions:
            we = vertices[region]
            plt.fill(*zip(*we),"white", alpha=0.3, edgecolor='black')
            
        plt.xlim(min(lons), max(lons))
        plt.ylim(min(lats), max(lats))
        plt.imshow(img, alpha=0.8, extent=[lons[0],lons[2],lats[0], lats[2]])
#        plt.scatter(dataset['X'], dataset['Y'], alpha=1, c='blue', s=2,edgecolor='')
        for i in range(len(centers)):
            plt.scatter(dataset.loc[dataset['Cluster'] == chr(i+65), 'X'],
                        dataset.loc[dataset['Cluster'] == chr(i+65), 'Y'],
                        s = 10, c = z[i%(len(z)-1)],label = 'Cluster ' + chr(65+i),
                        edgecolor ='')
        
#            i = 65
        for point in centers:
            plt.scatter(point[0], point[1], s=50, c='yellow',
                        edgecolor = 'black')
#                plt.text(point[0], point[1], chr(i) , fontdict=font, ha = 'right')
#                i += 1
        
        hauptbuhne = [8.3737641,48.99804315]    
        plt.scatter(hauptbuhne[0],hauptbuhne[1], s=60, c='black')
        plt.legend(markerscale=3)
        plt.show()
        if (plot):
            if years=='2017':
                plt.savefig(clustering +' clustering 2017.png')
            elif years=='2018':
                plt.savefig(clustering +' clustering 2018.png')
            else:
                plt.savefig(clustering +' clustering 2017-2018.png')
            
        return dataset, pois_to_clusters
    
        
#==============================================================================
# apo do kai kato einai i thliveri proseggisi per user! diladi me paths twn users
#==============================================================================
    #fill Nan with:(#df can be .csv or pandas dataframe )   
    def fillnanwith(self, df = 'filled_guest_trajectory_dataset.csv', method = 'none'):
        import pandas as pd
        if ".csv" in df:
            data = pd.read_csv(df).iloc[:,1:]# because CSVs have also stored indexes as an extra column at the begining
        else:
            data = df
        clusters = self.clusters
        data = data.dropna(how = 'all', axis = 1)
        #nan = 0:
        if (method == 'zeros'):
            data = data.fillna(0)
        
        #previous: (I fill every Nan with the first NEXT valid observation.The last ones that remain NAN at the end i use the previous last valid observation to fill them)
        elif (method == 'bfill'):
            data = data.fillna(method='bfill')
            data = data.fillna(method = 'ffill')
        
        #next:(inverse the method upon)
        elif (method == 'ffill'):
            data = data.fillna(method='ffill')
            data = data.fillna(method='bfill')
        
        #most frequent cluster of each guest
        elif (method == 'colfr'):
            data = data.fillna(data.mode().iloc[0]) 
                 
        #most frequent cluster in general
        elif (method == 'allfr'):
            iterate = range(len(data.columns))
            s = pd.Series()
            for i in iterate[(clusters+2):]:    
                y = data.iloc[:,i].value_counts()
                y.sort_index()
                s = s.add(y,fill_value = 0)
                s.sort_index()
    #            if (i==8 or i==9 or i==10 or i==11):  #check
    #                print("y",y)
    #                print("s",s)
            maxcluster = s.idxmax()
            data = data.fillna(maxcluster)
        elif (method == 'none'):
            pass
        else:
            print("please input something between 'zeros', 'bfill', 'ffill', 'colfr', 'allfr', 'none'") 
        return data
        
    def construct_dataset(self, data, final1, filtering_period):
        import pandas as pd
#        ##convert distributions from numbers to percentage##
#        columns = list(final1.columns[1:-1].values)
#        for column in columns:
#            final1[column] = final1[column]/final1['Total users']
#        final1 = final1.fillna(0)
#        final1 = final1.round(2)
#        final1['Total users'] = final1['Total users'].astype(int)
        
        ##new formation of dataset / add each user columns##
        dataset = final1
        cols1 = list(dataset.columns.values)
        #i want period to remain the first element so i do not sort it:
        k = cols1[1:] 
        # i only sort cluster names:
        k.sort() 
        #I keep all the DIFFERENT IDs that appear in the dataset in a list so as to later column them to dataframe
        user_unique_IDs = list(data.ID.unique())
        # I also sort the IDs but only between them. I want them at the end of the frame all together in order
        user_unique_IDs.sort()
        # I extend my dataframe to also contain columns for every diffrent guest ID in the desired order:
        dataset = pd.concat([dataset, pd.DataFrame(columns = user_unique_IDs, index = range(0,len(dataset)))])
        sorted_cols = [cols1[0]] + k + user_unique_IDs
        dataset = dataset[sorted_cols]
        dataset = dataset.iloc[:len(final1)]
        #put the cluster letter to the correct entry of the dataframe according to "data" dataframe.
        us = data['ID'].values
        tp = data['Period'].values
        cl = data['Cluster'].values
        #counters to reduce iterations and speed up the below loop execution
        counter = 0
        counter2 = 0
        
        for j in range(counter, len(dataset)):
            counter =+ 1
            for i in range (counter2, len(data)):
                if tp[i] ==  dataset.loc[j,'Period']:
                    dataset.loc[j,us[i]] = cl[i]
                    counter2 += 1
        #time_filtering
        if len(filtering_period) == 4:
            from defs import df_filtering
            filterer = df_filtering('non_filled_guest_trajectory_dataset.csv')
            filtered1 = filterer.df_column_bounding('Period','2017-07-21 ' + filtering_period[0] + ' to ' '2017-07-21 ' + filtering_period[1],'2017-07-21 ' + filtering_period[2] + ' to ' '2017-07-22 ' + filtering_period[3])
            filtered2 = filterer.df_column_bounding('Period','2017-07-22 ' + filtering_period[0] + ' to ' '2017-07-22 ' + filtering_period[1],'2017-07-22 ' + filtering_period[2] + ' to ' '2017-07-23 ' + filtering_period[3])
            filtered3 = filterer.df_column_bounding('Period','2017-07-23 ' + filtering_period[0] + ' to ' '2017-07-23 ' + filtering_period[1],'2017-07-23 ' + filtering_period[2] + ' to ' '2017-07-24 ' + filtering_period[3])
            frames = [filtered1,filtered2,filtered3]
            dataset = pd.concat(frames, ignore_index = 'True')
        return dataset
        
#creates obj of the class timePeriods and automatically calls all its method in series.
def preprocessor(cluster_number = 6 , timestep = 15, method = 'allfr' , min_users_per_period = 0, filtering_period = []):
    preprocessor = clustersPeriods(cluster_number, timestep)
    data = preprocessor.assign_users_in_regions() # data = pd.read_csv(dataframe_teliko.csv)
    final1 = preprocessor.cluster_distribution_per_period(data, min_users_per_period = min_users_per_period)
    dataset = preprocessor.construct_dataset(data, final1, filtering_period)
    if (len(method) > 0) or (method != 'none'):
        filled_dataset = preprocessor.fillnanwith(dataset, method = method)
    return (dataset,filled_dataset)

#time_filtering of dataset with column name = "Period"
def fltr(df, t11 = '05:00:00', t12 = '05:15:00', t21 = '23:45:00', t22 = '00:00:00',data2018=False):
    import pandas as pd
    if ".csv" in df:
        data = pd.read_csv(df).iloc[:,1:]# because CSVs have also stored indexes as an extra column at the begining
    else:
        data = df
    from datetime import datetime
    t1 = datetime.strptime(t11, '%H:%M:%S')
    t2 = datetime.strptime(t12, '%H:%M:%S')
    minutes = t2.minute-t1.minute
    filtering_period = [t11,t12,t21,t22]
    from defs import df_filtering
    filterer = df_filtering(data)
    #2017
    last=22 if t22=="00:00:00" else 21
    filtered1 = filterer.df_column_bounding('Period','2017-07-21 ' + filtering_period[0] + ' to ' '2017-07-21 ' + filtering_period[1],'2017-07-21 ' + filtering_period[2] + ' to ' '2017-07-'+str(last)+' ' + filtering_period[3])
    dates_column1 =  pd.Series(pd.date_range(start='2017-07-21 '+filtering_period[0], end='2017-07-21 ' + filtering_period[2], freq=str(minutes)+'min'))
    last+=1
    filtered2 = filterer.df_column_bounding('Period','2017-07-22 ' + filtering_period[0] + ' to ' '2017-07-22 ' + filtering_period[1],'2017-07-22 ' + filtering_period[2] + ' to ' '2017-07-'+str(last)+' ' + filtering_period[3])
    dates_column2 =  pd.Series(pd.date_range(start='2017-07-22 '+filtering_period[0], end='2017-07-22 ' + filtering_period[2], freq=str(minutes)+'min'))
    last+=1
    filtered3 = filterer.df_column_bounding('Period','2017-07-23 ' + filtering_period[0] + ' to ' '2017-07-23 ' + filtering_period[1],'2017-07-23 ' + filtering_period[2] + ' to ' '2017-07-'+str(last)+' ' + filtering_period[3])
    dates_column3 =  pd.Series(pd.date_range(start='2017-07-23 '+filtering_period[0], end='2017-07-23 ' + filtering_period[2], freq=str(minutes)+'min'))
    #2018 fest happened at 20-21-22/7
    last=21 if t22=="00:00:00" else 20
    filtered4 = filterer.df_column_bounding('Period','2018-07-20 ' + filtering_period[0] + ' to ' '2018-07-20 ' + filtering_period[1],'2018-07-20 ' + filtering_period[2] + ' to ' '2018-07-'+str(last)+' ' + filtering_period[3])
    dates_column4 =  pd.Series(pd.date_range(start='2018-07-20 '+filtering_period[0], end='2018-07-20 ' + filtering_period[2], freq=str(minutes)+'min'))
    last+=1
    filtered5 = filterer.df_column_bounding('Period','2018-07-21 ' + filtering_period[0] + ' to ' '2018-07-21 ' + filtering_period[1],'2018-07-21 ' + filtering_period[2] + ' to ' '2018-08-'+str(last)+' ' + filtering_period[3])
    dates_column5 =  pd.Series(pd.date_range(start='2018-07-21 '+filtering_period[0], end='2018-07-21 ' + filtering_period[2], freq=str(minutes)+'min'))
    last+=1
    filtered6 = filterer.df_column_bounding('Period','2018-07-22 ' + filtering_period[0] + ' to ' '2018-07-22 ' + filtering_period[1],'2018-07-22 ' + filtering_period[2] + ' to ' '2018-07-'+str(last)+' ' + filtering_period[3])
    dates_column6 =  pd.Series(pd.date_range(start='2018-07-22 '+filtering_period[0], end='2018-07-22 ' + filtering_period[2], freq=str(minutes)+'min'))

    if data2018:
        frames=[filtered1,filtered2,filtered3,filtered4,filtered5,filtered6]
        dates_column = pd.concat([dates_column1,dates_column2,dates_column3,
                                  dates_column4,dates_column5,dates_column6])
    else:
        frames = [filtered1,filtered2,filtered3]        
        dates_column = pd.concat([dates_column1,dates_column2,dates_column3])
    data = pd.concat(frames, ignore_index = 'True')
#    data = pd.concat([data,dates_column],axis=1)
    
    return data,dates_column