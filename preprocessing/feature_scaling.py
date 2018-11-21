# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 19:48:04 2018

@author: spele
"""
import numpy as np
def feature_scaling(final2, cltopred, predictor_cols, mini, maxi,encode_time_indexes,artists):
    from sklearn.preprocessing import MinMaxScaler
    import pandas as pd
    sc_pop = 0
    sc_cl = 0
    sc_tot = 0
    if cltopred in predictor_cols:
        sc_cl = MinMaxScaler(feature_range=(mini, maxi))
        final2[cltopred] = sc_cl.fit_transform(final2[cltopred].values.reshape(-1,1))
    if isinstance(artists, pd.DataFrame):
        #popularity feature scaliNg
        sc_pop = MinMaxScaler(feature_range=(mini, maxi))
        sc_pop.fit(pd.concat([artists.iloc[:,0], artists.iloc[:,1]]).values.reshape(-1, 1))
        final2[cltopred+"pop"] = sc_pop.transform(final2[cltopred+"pop"].values.reshape(-1,1))
    if "Total users" in predictor_cols:
        sc_tot = MinMaxScaler(feature_range=(mini, maxi))
        final2["Total users"] = sc_tot.fit_transform(final2["Total users"].values.reshape(-1,1))
    
    dataset = final2.values
    
    # I scale time indexes only in the case they re not label encoded
    sc_index=0
    sc_day=0
    if "Time index" in predictor_cols:
#        if (encode_time_indexes==False):
        sc_index = MinMaxScaler(feature_range=(mini, maxi))
        time_index_cols = sc_index.fit_transform(final2["Time index"].values.reshape(-1,1))
#        else:
#            from sklearn import preprocessing
#            lb = preprocessing.LabelBinarizer()
#            time_index_cols = lb.fit_transform(final2["Time index"])
#            time_index_cols = time_index_cols[:,1:] #avoid dummy variable trap
        final2.drop(columns = "Time index", inplace=True)
        dataset = np.concatenate([final2.values, time_index_cols], axis = 1)
    if "Day index" in predictor_cols:
        if (encode_time_indexes==False):
            sc_day = MinMaxScaler(feature_range=(mini, maxi))
            day_index_cols = sc_day.fit_transform(final2["Day index"].values.reshape(-1,1))
        else:
            from sklearn import preprocessing
            lb = preprocessing.LabelBinarizer()
            day_index_cols = lb.fit_transform(final2["Day index"])
            day_index_cols = day_index_cols[:,1:] #avoid dummy variable trap
        final2.drop(columns = "Day index", inplace=True)
        dataset = np.concatenate([final2.values, day_index_cols], axis = 1)
    return dataset, sc_cl, sc_pop, sc_tot, sc_index, sc_day