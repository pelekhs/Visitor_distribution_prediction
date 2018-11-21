# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 18:24:17 2018

@author: spele
"""

#FROM 1 MINUTE TO 30 MINUTES

def theregulator(final2, minutes, multiplier,clwithlive):
#final2: the full feature dataset A,B,pops,temp,cond, time index
#multiplier: new timestep is multiplier times bigger than the final2 one
#minutes = time periods length of final2 in minutes
    import numpy as np
    import pandas as pd

    
    a,b,c,d,e,f,pop1,pop2,tot,temp,cond,time,ts = [],[],[],[],[],[],[],[],[],[],[],[],[]
    for i in final2.index.values[final2.index.values % multiplier == 0]:
        a.append(np.mean(final2['A'].values[i:i+multiplier]))
        b.append(np.mean(final2['B'].values[i:i+multiplier]))
        c.append(np.mean(final2['C'].values[i:i+multiplier]))
        d.append(np.mean(final2['D'].values[i:i+multiplier]))
        e.append(np.mean(final2['E'].values[i:i+multiplier]))
        f.append(np.mean(final2['F'].values[i:i+multiplier]))
        ts.append(final2['Time series'].values[i])
        pop1.append(np.mean(final2[i:i+multiplier][clwithlive[0]+'pop']))
        pop2.append(np.mean(final2[i:i+multiplier][clwithlive[1]+'pop']))
        tot.append(np.mean(final2[i:i+multiplier]['Total users']))
        temp.append(np.mean(final2[i:i+multiplier]['Temperature']))
        cond.append(np.mean(final2[i:i+multiplier]['Conditions']))
        time.append(np.mean(final2[i:i+multiplier]['Time index']))
    time = (np.asarray(time) // multiplier).astype(int)
    final_multi = pd.DataFrame({'A': np.asarray(a),'B': np.asarray(b),'C': np.asarray(c),'D': np.asarray(d),'E': np.asarray(e),
                               'F': np.asarray(f), clwithlive[0]+'pop': np.asarray(pop1), clwithlive[1]+'pop': np.asarray(pop2),
                               'Total users': np.asarray(tot),'Temperature': np.asarray(temp), 'Conditions': np.asarray(cond),
                                'Time index': np.asarray(time), 'Time series':np.asarray(ts)})
    return final_multi