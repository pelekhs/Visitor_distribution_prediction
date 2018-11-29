# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 16:35:56 2018

@author: spele
"""

def weather(final1):
    import pandas as pd
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
    
    return df