# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 14:59:53 2021

@author: fdost
"""

import time
import os
import requests
from PIL import Image
import subprocess
from multiprocessing import Pool

from discogsApi import getReleaseInfo

import database

imDir = 'images'
os.makedirs(imDir, exist_ok=True)

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
    imPath = os.path.join(imDir, str(idNum)+'.jpg')
    if os.path.exists(imPath):
        return False
    
    release = database.getRelease(idNum)
    
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
    while not result:
        r = subprocess.run(dlCmd)
        if r.returncode == 0:
            result = True
        else:
            result = False
            time.sleep(10)

def downloadImageTry(idNum):
    try:
        downloadImage(idNum)
    except Exception as e:
        print(idNum, e)

def removeImageIfBad(idNum):
    imPath = os.path.join(imDir, str(idNum) + '.jpg')
    if not os.path.exists(imPath):
        return False
    try:
        Image.open(imPath)
    except:
        print("Removing "+imPath)
        os.remove(imPath)
    
def getAllImages():
    releaseIds = database.getAllReleaseIds()
    print('{} entries'.format(len(releaseIds)))
    
    usePool = False
    
    if usePool:
        with Pool(2) as p:
            p.map(removeImageIfBad, releaseIds)
            p.map(downloadImageTry, releaseIds)
    else:
        for idNum in releaseIds:
            downloadImageTry(idNum)
            #time.sleep(1)
            
if __name__ == '__main__':
    getAllImages()