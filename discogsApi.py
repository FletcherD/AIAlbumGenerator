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
discogsApiSearch = discogsApiUrl + 'database/search?artist={}&release_title={}&type=release&per_page=50&' + discogsApiToken

class RateLimiter:
    def __init__(self):
        self.remaining = 60
        self.reset_time = None
    
    def wait_if_needed(self):
        # Be more conservative - wait when we have 5 or fewer requests left
        if self.remaining <= 5:
            wait_time = min(60, 10)  # Wait up to 60 seconds, start with 10
            print(f"  Rate limit low ({self.remaining} remaining), waiting {wait_time}s...")
            time.sleep(wait_time)
    
    def update_from_headers(self, headers):
        if 'x-discogs-ratelimit-remaining' in headers:
            self.remaining = int(headers['x-discogs-ratelimit-remaining'])
            #print(f"Rate limit: {self.remaining} requests remaining")

rate_limiter = RateLimiter()

def apiGet(url):
    rate_limiter.wait_if_needed()
    
    r = requests.get(url)
    rate_limiter.update_from_headers(r.headers)
    
    data = json.loads(r.text)
    
    # Handle specific rate limit response
    if 'message' in data and 'rate limit' in data['message'].lower():
        print("Hit rate limit, waiting 60 seconds...")
        time.sleep(60)
        rate_limiter.remaining = 60  # Reset after waiting
        r = requests.get(url)
        rate_limiter.update_from_headers(r.headers)
        data = json.loads(r.text)
    
    return data

def getReleaseInfo(releaseNum):
    releaseUrl = discogsApiReleases.format(releaseNum)

    data = apiGet(releaseUrl)

    while 'message' in data and 'quickly' in data['message']:
        print(releaseNum, data)
        time.sleep(10)
        data = apiGet(releaseUrl)

    #print(r.text)
    return data


def getReleasePopularity(release):
    return release['community']['have']


def getReleaseArtist(r):
    artistName = r['title'].split(' - ')[0]
    artistName = re.sub(' \([0-9]*\)', '', artistName)
    return artistName


def getArtistPopularity(artistName):
    searchUrl = discogsApiArtist.format(urllib.parse.quote(artistName))
    data = apiGet(searchUrl)
    releases = data['results']
    releases = [r for r in releases if getReleaseArtist(r).lower().startswith(artistName.lower())]
    popularity = 0
    for r in releases:
        popularity += getReleasePopularity(r)
    return popularity


def searchReleaseByArtistAndTitle(artistName, albumTitle):
    """
    Search for a release on Discogs by artist name and album title.
    
    Args:
        artistName (str): The name of the artist
        albumTitle (str): The title of the album
        
    Returns:
        dict: Search results from Discogs API, or None if no results found
    """
    searchUrl = discogsApiSearch.format(urllib.parse.quote(artistName), urllib.parse.quote(albumTitle))
    
    data = apiGet(searchUrl)
    
    # Handle rate limiting
    while 'message' in data and 'quickly' in data['message']:
        print(f"Rate limited when searching for {artistName} - {albumTitle}, waiting...")
        time.sleep(10)
        data = apiGet(searchUrl)
    
    if 'results' in data and len(data['results']) > 0:
        return data['results']
    else:
        return None
