#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 16:30:15 2018

@author: anastasis
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 17:53:24 2018

@author: anastasis
"""
def create_artist(timestep = 15, df="pois_toclusters.csv", only_start = 'No'):
    import pandas as pd
    from datetime import timedelta, datetime
    import numpy as np
    if ".csv" in df:
        my_df_ex1 = pd.read_csv(df)
    else: 
        my_df_ex1 = df 
    metrics = pd.read_excel('metrs.xls', header= None, names = ['artist', 'metr', 'rep'])
    
    
    
    start = my_df_ex1['Date'].iloc[0]+' 00:00:00'
    start = datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
    end = my_df_ex1['Date'].iloc[0]+' 23:59:59'
    end = datetime.strptime(end, '%Y-%m-%d %H:%M:%S')
    
    periods = []
    periods_to_numbers = []
    #list of each period in from-to view
    periods_from_to = []
    #same as periods_to_numbers but instead of numbers we got from-to
    each_period = []
    
    def check_user(Timestamp):
        for i in range(len(periods)):
            if (Timestamp >= periods[i] and Timestamp < periods[i+1]):
                periods_to_numbers.append(i)
                each_period.append(periods_from_to[i])
                break
            
    
    #def period(time):
    while start < end:
        start1 = start
        start = start + timedelta(minutes=timestep)
        periods.append(start1.strftime('%H:%M:%S'))
        periods_from_to.append(start1.strftime('%H:%M:%S')+' to '+start.strftime('%H:%M:%S'))
    
    periods.append('23:59:59')

    for timestamp in my_df_ex1['Time']:    
        check_user(timestamp)
    
    #ta 3 pou vgainoun ap e3w einai komple
    my_df_ex1['Period'] = each_period
#    trelakiko = trelakiko.copy()
    each_period = list(zip(my_df_ex1.Date, my_df_ex1.Period))
    my_df_ex1['Period'] = each_period    
    my_df_ex1.set_index('Period', inplace = True)


#Μεχρι αυτο το σημειο απλα εφτιαξα τα time periods και τιποτα περισσοτερο. Απο δω και περα γινεται
#η αντιστοιχιση καλλιτερχνων-περιοδων-μετρικων κοκ.

#(1) φτιαχνω λιστα με τα stages και βρισκω σε ποια χρονικη περιοδο ξεκιναει ο καθε καλλιτεχνης. Οποτε εχω ενα dict στο οποιο σαν
#    keys ειναι τα stages και μεσα περιεχονται ολα τα periods και τα periods που ΞΕΚΙΝΑΕΙ ο καθε καλλιτεχνης.
    
    b = list(my_df_ex1.groupby('Buhne').groups.keys())
    
    stages = {}
    
    for letters in b:
        stages.update({letters: my_df_ex1[my_df_ex1['Buhne'] == letters]})
    
    a = my_df_ex1.groupby(['Date']).size().index.tolist()
#    a.extend(('2018-07-20', '2018-07-21', '2018-07-22'))
    import itertools
    periods_from_to = list((itertools.product(a, periods_from_to)))


    def assign_to_per(art_clust):
        trelakiko = pd.DataFrame(index = periods_from_to, columns = ['Artist_1'])
        for periods in periods_from_to:
            try:
                trelakiko.at[[periods], 'Artist_1'] = art_clust.loc[[periods]]['Artist'][0]
#            except AttributeError:
#                trelakiko.set_value([periods], 'Artist_1', art_clust.loc[[periods]]['Artist'][0])
            except KeyError:
                pass
        return trelakiko

    trelakiko = {}
    
    for letters in b:
        trelakiko.update({letters: assign_to_per(stages[letters])})        

#(2) Εδω φιλαρω τη λιστα με τη συνολικη διαρκεια που παιζει ο καθε καλλιτεχνης. Συγκεκριμενα θελουμε σε καθε 
#    περιπτωση να υπαρχει ενα time period κενο μεταξυ 2 διαφορετικων καλλιτεχνω στο ιδιο stage. Επισης ολα τα live
#    τελειωνουν στις 12 το βραδυ (00:00:00).
    if only_start == 'No':
        def copy_artists_periods(buhne):
            for artists in buhne['Artist_1'][buhne['Artist_1'].isnull() == False]:
                ind = buhne.index
                i = ind.get_loc(buhne.index[buhne['Artist_1'] == artists].get_values()[0])
                k = '00:00:00' in buhne.iloc[i].name
                l = 0
                while (i < len(buhne)-2 and (buhne.iloc[i+1].isnull()[0]) == True and k == False and l<105//timestep):
                    buhne.iloc[i+1]['Artist_1'] = buhne.iloc[i]['Artist_1']
                    i += 1
                    l += 1
                    k = '23:30:00' in buhne.iloc[i].name[1]
            return buhne
                    
        
        final_list = {}
        for letters in b:
            final_list.update({letters: copy_artists_periods(trelakiko[letters])})
    else:
        final_list = trelakiko
#(3) Εδω με εναν αξιοσημειωτο τροπο βγαλμενο απο ταινια αντιστοιχιζω τα stages με τα clusters στα οποια βρισκονται.
#    Για καποιο λογο δεν μπορουσα να σκεφτω πιο απλο τροπο να κανω το mapping.

    artists = pd.DataFrame(index = periods_from_to, columns = b)
    for keys in final_list:
        artists[keys] = final_list[keys].values
        
    clusters = {}
    
    for letters in b:
        clusters.update({letters: my_df_ex1['Cluster'][my_df_ex1['Buhne'] == letters].unique()[0]})

#(4) Εδω γινεται το rename των στηλων συμφωνα με το mapping που εφτιαξα παραπανω
    artists = artists.rename(columns=clusters)

#(5) Προσθετω στις μετρικες το τι γινεται στην περιπτωση του NaN, δηλαδη σε τι αντιστοιχιζεται η κατασταση NaN.
    metrics = metrics.append(pd.DataFrame([[np.NaN, -5, 'NO_SHOW']], columns = ['artist', 'metr', 'rep']))
    
#(6) Βαζω σαν index τους artists ετσι ωστε να μπορω να κανω το mapping.
    metrics.set_index('artist', inplace=True)
    
#(7) Κανω την αντιστοιχιση artists ---> metrics
    artists = artists.applymap(lambda x: metrics.loc[x]['metr'])
    
#(7) Κάνω combine ολες τις στηλες που εχουν το ιδιο ονομα και τις βαζω σε εναν scaler έτσι ώστε να εχω σαν 0 τις 
#    περιοδους που δεν υπάρχει live και να μπορω να προσθεσω τις μετρικές των live για τις ιδιες περιοδους στο ιδιο κλαστερ.
#    Τωρα ολες οι τιμες ειναι θετικες οποτε μπορουν να προστεθουν. Πλεον οταν σ ενα κλαστερ υπαρχει ΕΝΑ live σε μια χρονικη
#    περιοδο (ενω σε αλλες υπαρχουν 2 πχ) θα προστεθει η μετρικη του καλλιτεχνη με το 0 (δλδ οχι 2ρο live) οποτε και θα
#    παρουμε σα συνολικο αποτελεσμα την μετρικη του ενος καλλιτεχνη.
#    
#    ΕΤΣΙ ΒΕΒΑΙΑ ΜΠΟΥΣΤΑΡΟΝΤΑΙ ΚΑΤΑ ΠΟΛΥ ΟΙ ΜΕΤΡΙΚΕΣ ΣΕ ΜΙΑ ΠΕΡΙΟΔΟ ΠΟΥ ΠΑΙΖΟΥΝ 2 ΚΑΛΛΙΤΕΧΝΕΣ ΤΑΥΤΟΧΡΟΝΑ ΣΤΟ ΙΔΙΟ ΚΛΑΣΤΕΡ!!!!
    
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(artists.values)
    
    artists[list(artists.columns.unique())] = scaler.fit_transform(artists[list(artists.columns.unique())])
    artists = artists.groupby(artists.columns, axis=1).max()
    if only_start=='end':
        artists.iloc[:,:] = np.roll(artists.iloc[:,:],-1,0)
    return artists