#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 17:53:24 2018

@author: anastasis
"""
def create(timestep):
    from my_df import my_df_ex1
    from datetime import timedelta, datetime
    
    start = '2017-07-21 00:00:00'
    start = datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
    end = '2017-07-21 23:59:59'
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
    
    #if __name__ == '__main__' :
    for timestamp in my_df_ex1['Time']:    
        check_user(timestamp)
    
    #ta 3 pou vgainoun ap e3w einai komple
    my_df_ex1['Period'] = each_period
    trelakiko = my_df_ex1[0:0]
#    trelakiko = trelakiko.copy()
    
#    def check_same_users(period):
##        global trelakiko
#        my_df_ex = my_df_ex1[my_df_ex1['Period'] == period]
#        my_df_ex = my_df_ex.drop_duplicates(subset='ID', keep='first', inplace = False)
#        trelakiko.append(my_df_ex)
    
    for periods in periods_from_to:
        my_df_ex = my_df_ex1[my_df_ex1['Period'] == periods]
        my_df_ex = my_df_ex.drop_duplicates(subset='ID', keep='first', inplace = False)
        trelakiko = trelakiko.append(my_df_ex)
    
    trelakiko.sort_values(by = 'Timestamp', inplace = True)
    trelakiko = trelakiko.reset_index(drop=True)
    
    trelakiko.to_csv('dataframe_1.csv')
    return periods_from_to