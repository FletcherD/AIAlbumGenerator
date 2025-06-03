# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 14:59:53 2021

@author: fdost
"""
import random
import time
import os
import requests
from PIL import Image
import subprocess
from multiprocessing import Pool

import DiscogsDataset

os.makedirs(DiscogsDataset.IMAGE_DIR_PATH_UNPROCESSED, exist_ok=True)


def getImageUrl(release):
    if 'images' in release:
        image = release['images'][0]
        url = image['resource_url']
        if len(url) == 0:
            return None
        return url
    else:
        return None


def downloadImage(idNum):
    imPath = os.path.join(DiscogsDataset.IMAGE_DIR_PATH_UNPROCESSED, str(idNum)+'.jpg')
    if os.path.exists(imPath):
        return False
    
    release = DiscogsDataset.getRelease(idNum)
    
    url = getImageUrl(release)
    # if url is None:
    #     releaseNew = getReleaseInfo(release['id'])
    #     if 'message' in releaseNew and 'not found' in releaseNew['message']:
    #         print('Release {} no longer exists'.format(release['id']))
    #         del dataset[release['id']]
    #         return False
    #     release = releaseNew
    #     dataset[release['id']] = release
    #     url = getImageUrl(release)
    if url is None:
        return False
    
    print(release['id'], url)

    dlCmd = ['yt-dlp', '-q', url, '-o', imPath]
    result = False
    #while not result:
    r = subprocess.run(dlCmd)
    if r.returncode == 0:
        result = True
    else:
        result = False
        print(r.returncode)
        time.sleep(30)


def downloadImageTry(idNum):
    try:
        downloadImage(idNum)
    except Exception as e:
        print(idNum, e)


def removeImageIfBad(idNum):
    imPath = os.path.join(DiscogsDataset.IMAGE_DIR_PATH_UNPROCESSED, str(idNum) + '.jpg')
    if not os.path.exists(imPath):
        return False
    try:
        Image.open(imPath)
    except:
        print("Removing "+imPath)
        #os.remove(imPath)


def getAllImages():
    releaseIds = DiscogsDataset.getAllReleaseIds()
    print('{} entries'.format(len(releaseIds)))
    random.shuffle(releaseIds)
    
    if N_WORKERS > 1:
        with Pool(N_WORKERS) as p:
            #p.map(removeImageIfBad, releaseIds)
            p.map(downloadImageTry, releaseIds)
    else:
        for idNum in releaseIds:
            #removeImageIfBad(idNum)
            downloadImageTry(idNum)
            #time.sleep(1)

N_WORKERS = 1
            
if __name__ == '__main__':
    getAllImages()
