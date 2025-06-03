# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import requests
import json
import random
import time
import urllib
import re
import os

from dotenv import load_dotenv
load_dotenv()

myToken = os.getenv('DISCOGS_TOKEN')

discogsApiUrl = 'https://api.discogs.com/'
discogsApiToken = 'token=' + myToken
discogsApiReleases = discogsApiUrl + 'releases/{}?' + discogsApiToken
discogsApiArtist = discogsApiUrl + 'database/search?artist={}&type=release&per_page=100&' + discogsApiToken


def getReleaseInfo(releaseNum):
    releaseUrl = discogsApiReleases.format(releaseNum)
    
    r = requests.get(releaseUrl)
    data = json.loads(r.text)
    
    while 'message' in data and 'quickly' in data['message']:
        print(releaseNum, data)
        time.sleep(10)
        r = requests.get(releaseUrl)
        data = json.loads(r.text)
        
    #print(r.text)
    return data


def getReleasePopularity(release):
    return release['community']['have']


def getReleaseArtist(r):
    artistName = r['title'].split(' - ')[0]
    artistName = re.sub(' \([0-9]*\)', '', artistName)
    return artistName


def getArtistPopularity(artistName):
    apiSearchUrl = discogsApiArtist.format(urllib.parse.quote(artistName))
    r = requests.get(apiSearchUrl)
    data = json.loads(r.text)
    releases = data['results']
    releases = [r for r in releases if getReleaseArtist(r).lower().startswith(artistName.lower())]
    popularity = 0
    for r in releases:
        popularity += getReleasePopularity(r)
    return popularity
