import requests
import json
import random
import time
import pickle
import shutil
import os

import DiscogsDataset

url = r'https://api.discogs.com/releases/'

token = os.getenv('DISCOGS_TOKEN')

maxNum = 32000000

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
    num_added = 0
    while num_added < 1000:
        rNum = random.randint(0, maxNum)
        while DiscogsDataset.getRelease(rNum) is not None:
            rNum = random.randint(0, maxNum)
        
        try:
            rData = getReleaseInfo(rNum)
            
            if 'title' in rData:
                print('{}: {} - {}'.format(rNum, rData['artists_sort'], rData['title']))
                DiscogsDataset.addRelease(rNum, rData)

            num_added += 1

        except Exception as e:
            print(e)
            
