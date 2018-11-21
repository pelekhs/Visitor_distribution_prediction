import pandas as pd
import time

# Importing the dataset
dataset = pd.read_json('http://www.yellowmap.de/Presentation/BASMATI/TESTING_Basmati-Track_2017_2018.json', lines=True) 

#'null' filtering
dataset1 = dataset[dataset.astype(str)['Points'] != '[]']
dataset1 = dataset1.reset_index(drop=True)

ts = []
tsgmt = []
x = []
y = []
dt = []
idnumb = []
oid = []
tm = []
rows = len(dataset1)


for i in range(rows):
    man = dataset1.iloc[i].get(0)
    man1 = dataset1.iloc[i].get(1)
    flag = False
    ti = []
    xi = []
    yi = []
    tigmt = []
    dti = []
    tmi = []
    for j in range(len(man)):
        if  ((int(str(man[j][2])[0:10]) > 1500595200) & (int(str(man[j][2])[0:10]) < 1500854399)):
            yi.append(man[j][0])
            xi.append(man[j][1])
            ti.append(int(str(man[j][2])[0:10]))
            tigmt.append(time.strftime('%Y-%m-%d %H:%M:%S' , time.gmtime(ti[-1])))
            dti.append(time.strftime(('%Y-%m-%d'), time.gmtime(ti[-1])))
            tmi.append(time.strftime(('%H:%M:%S'), time.gmtime(ti[-1])))
            flag = True
#temporary lists xi, yi, ti that are created in the j loop if condition is true
#we append them to x,y,ts lists if the condition is true (flag) and then they re 
#flushed again for the next row. These lists contain all the successive xi, ti, tis
#that are included in each session so that we dont miss any useful data. x,y,ts
#are from now on lists of lists (2d arrays) but we can treat them as simple lists 
    if (flag):
        idnumb.append(man1['UserId'])
        oid.append(dataset1.iloc[i].get(2)['$oid'])
        y.append(yi)
        x.append(xi)
        ts.append(ti)
        tsgmt.append(tigmt)
        dt.append(dti)
        tm.append(tmi)


d = { '$oid':oid, 'ID':idnumb, 'Epoch': ts, 'Timestamp':tsgmt, 'Date':dt, 'Time': tm, 'X':x,'Y':y}
my_df = pd.DataFrame(data = d)

#null filtering for empty IDs
my_df = my_df[my_df.astype(str)['ID'] != '']
my_df = my_df.reset_index(drop=True)


#create alternative dataset my_df_ex converting the lists in new columns for easier accessing of X,Y,t in the future
ts = []
tsgmt = []
x = []
y = []
dt = []
idnumb = []
oid = []
tm = []
rows = len(dataset1)

for i in range(rows):
    man = dataset1.iloc[i].get(0)
    man1 = dataset1.iloc[i].get(1)
    for j in range(len(man)):
        if  (((int(str(man[j][2])[0:10]) > 1500595200) & (int(str(man[j][2])[0:10]) < 1500854399)) or ((int(str(man[j][2])[0:10]) > 1532044800) & (int(str(man[j][2])[0:10]) < 1532304000))):
            idnumb.append(man1['UserId'])
            oid.append(dataset1.iloc[i][2]['$oid'])
            y.append(man[j][0])
            x.append(man[j][1])
            ts.append(int(str(man[j][2])[0:10]))
            tsgmt.append(time.strftime('%Y-%m-%d %H:%M:%S' , time.gmtime(ts[-1])))
            dt.append(time.strftime(('%Y-%m-%d'), time.gmtime(ts[-1])))
            tm.append(time.strftime(('%H:%M:%S'), time.gmtime(ts[-1])))
        
d = { '$oid':oid, 'ID':idnumb, 'Epoch': ts, 'Timestamp':tsgmt, 'Date':dt, 'Time': tm, 'X':x,'Y':y}
my_df_ex = pd.DataFrame(data = d)

#null filtering for empty IDs
my_df_ex = my_df_ex[my_df_ex.astype(str)['ID'] != '']
my_df_ex = my_df_ex.reset_index(drop=True)
#my_df_ex.to_csv('dataframe.csv')