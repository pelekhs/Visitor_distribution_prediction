# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 16:46:38 2018

@author: spele
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

final2 = pd.read_csv('final2_15mins.csv')
x = np.arange(len(final2["Time index"]))
y1 = final2["Time index"]/max(final2["Time index"])
y2 = final2["Time index"].apply(lambda x: (np.sin(((2*np.pi*x)/(max(final2["Time index"]))))))
y3 = final2["Time index"].apply(lambda x: 0.5 + 0.5*(np.cos(((np.pi-2*np.pi*x)/(max(final2["Time index"]))))))

index=[x for x in final2['Time series']]
pt = [(datetime.strptime(index[x], '%Y-%m-%d %H:%M:%S')) for x in range(len(index))]

plt.figure('Alternative timestep represantations')
plt.subplot(3,1,1)
plt.plot(pt,y1)
plt.title('Linear')
plt.subplot(3,1,2)
plt.plot(pt,y2)
plt.title('Sinusoidal')
plt.subplot(3,1,3)
plt.plot(pt,y3)
plt.title('Cosinusoidal')
plt.show()

