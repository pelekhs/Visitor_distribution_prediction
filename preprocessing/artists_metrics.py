#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 16:58:46 2018

@author: anastasis
"""
import requests
import pandas as pd
from multiprocessing import Pool
import xlwt


#art_data = pd.read_csv('artists_list.csv')
art_data = pd.read_csv('artists_list_2018.csv')
#CSVFILE = 'metrs.xls'
CSVFILE = 'metrs_2018.xls'


reach = []
benchmark = []

def get_metr(artist):
    token = '38790b71deccc5c7f01c369de19ddee9'
    print(artist)
    s = requests.get('https://api.nextbigsound.com/search/v1/artists/?access_token='+token+'&fields=id,name,images,scores,category&limit=15&query='+str(artist))
    get_id = s.json()
    try:
        id_no = get_id['artists'][0]['id']
        
        r = requests.get('https://api.nextbigsound.com/metrics/v1/entity/'+str(id_no)+'/reach?access_token='+token+'&days=90')
        ans = r.json()
        metr = ans['overall']['score']
        
        
        r_2 = requests.get('https://api.nextbigsound.com/soti/v1/artist/'+str(id_no)+'/benchmark?access_token='+token)
        ans_2 = r_2.json()
        metr_2 = ans_2['artist']['stage']['name']
    except IndexError:
        metr = -2
        metr_2 = 'Undiscovered'    
    except TypeError:
        r = requests.get('https://api.nextbigsound.com/meta/v1/artists/'+str(id_no)+'?access_token='+token+'&fields=name,images,category,metadata,is_subscribed,scores')
        ans = r.json()
        metr = -1.8
        metr_2 = ans['scores']['stage']
        if metr_2 == None:
            metr_2 = 'Undiscovered'
        elif metr_2 == 'Promising':
            metr = -1
        elif metr_2 == 'Mainstream':
            metr = 0
    except KeyError:
        metr = -2
        metr_2 = 'Undiscovered'
    
    ans = [artist, metr, metr_2]

    
    return ans
#    return metr, metr_2


if __name__ == '__main__':
    pool = Pool(10)
    whatever = pool.map(get_metr, list(art_data['Artist']))
    
    book = xlwt.Workbook(encoding="utf-8")
    sheet1 = book.add_sheet("Sheet 1", cell_overwrite_ok=True)
    
    for i in range(len(whatever)):
        sheet1.write(i, 0, whatever[i][0])
        sheet1.write(i, 1, whatever[i][1])
        sheet1.write(i, 2, whatever[i][2])
    book.save(CSVFILE)
    
    #for artist in art_data['Artist']:
    #    print(artist)
    #    metr, metr_2 = get_metr(artist)
    #    reach.append(metr)
    #    benchmark.append(metr_2)
    
        
        