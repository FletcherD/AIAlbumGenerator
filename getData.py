# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import requests
import json
import random
import time
import pickle
import shutil

import database

url = r'https://api.discogs.com/releases/'
token = 'WttDSokZGdcaSNcgLeLSTYJRcZcXmYQCpYwkdCZh'

maxNum = 28696234

def getReleaseInfo(releaseNum):
    releaseUrl = url + str(releaseNum) + '?token='+token
    
    r = requests.get(releaseUrl)
    data = json.loads(r.text)
    
    while 'message' in data and 'quickly' in data['message']:
        print(releaseNum, data)
        time.sleep(10)
        r = requests.get(releaseUrl)
        data = json.loads(r.text)
        
    #print(r.text)
    return data

if __name__ == "__main__":
    for i in range(100000):
        rNum = random.randint(0, maxNum)
        while database.getRelease(rNum) is not None:
            rNum = random.randint(0, maxNum)
        
        try:
            rData = getReleaseInfo(rNum)
            
            if 'title' in rData:
                print('{}: {} - {}'.format(rNum, rData['artists_sort'], rData['title']))
                database.addRelease(rNum, rData)

        except Exception as e:
            print(e)
            