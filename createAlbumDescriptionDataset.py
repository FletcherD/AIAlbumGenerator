import math

import DiscogsDataset
import os
import csv
import re
import json
import random
import glob
import langdetect
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
from datasets import load_dataset

def getArtistStr(release):
    r = ''
    if 'artists' not in release:
        return None
    for artist in release['artists']:
        artistName = artist['name']
        artistName = re.sub(r' \([0-9]*\)', '', artistName)
        r += artistName
        if artist['join'] != '':
            if artist['join'] != ',':
                r += ' '
            r += artist['join'] + ' '
    return r

def getGenreStr(release):
    genreList = []
    if 'genres' in release:
        genreList += release['genres']
    if 'styles' in release:
        genreList += release['styles']
    else:
        genreList = ['None']
    genreSubstitutions = {'Folk, World, & Country': "Folk / World / Country"}
    for i, genre in enumerate(genreList):
        if genre in genreSubstitutions:
            genreList[i] = genreSubstitutions[genre]
    #genreList = set(genreList)
    return ', '.join(genreList)

def getReleaseDescriptionStr(release):
    r = ''
    r += 'Artist: {}\n'.format(getArtistStr(release))
    r += 'Title: {}\n'.format(release['title'])
    r += '{} - {}\n'.format(getArtistStr(release), release['title'])
    r += 'Genre: {}\n'.format(getGenreStr(release))
    r += 'Year: {}\n'.format(release['year'])
     for track in release['tracklist']:
         r += '\t{position}: {title}\n'.format(**track)
    return r


def createAlbumDescriptionDataset():
    with open('albumDescriptionDataset.jsonl', 'w') as f:
        for idNum in tqdm(releaseIds):
            try:
                release = DiscogsDataset.getRelease(idNum)
                releaseStr = getReleaseDescriptionStr(release)

                language = langdetect.detect(releaseStr)
                if language != 'en':
                    continue

                trainingStr = getReleaseDescriptionStr(release)
                trainingJson = {'note': '### '+trainingStr}
                f.write(json.dumps(trainingJson) + '\n')
            except Exception as e:
                print(e)
                print(idNum)


if __name__ == '__main__':
    createTextToImageDataset()

