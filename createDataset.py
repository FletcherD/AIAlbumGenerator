import database
import os
import re
import json
import random
import glob
import langdetect
from tqdm import tqdm

imDir = 'images'
metadataFilePath = os.path.join(imDir, 'metadata.jsonl')

prompt = """The album artwork for an album titled "{title}" \
by the artist {artistStr}, released in {year}, in the genres {genreStr}"""

bosToken = "<s>"
eosToken = '</s>'

def getArtistStr(release):
    r = ''
    if 'artists' not in release:
        return None
    for artist in release['artists']:
        artistName = artist['name']
        artistName = re.sub(' \([0-9]*\)', '', artistName)
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

def getPromptStr(release):
    return prompt.format(**release, artistStr=getArtistStr(release), genreStr=getGenreStr(release))

def getTrainingStr(release):
    r = 'Artist: {}\n'.format(getArtistStr(release))
    r += 'Title: {}\n'.format(release['title'])
    #r += '{} - {}\n'.format(getArtistStr(release), release['title'])
    r += 'Genre: {}\n'.format(getGenreStr(release))
    r += 'Year: {}\n'.format(release['year'])
    for track in release['tracklist']:
        r += '\t{position}: {title}\n'.format(**track)
    return r

def removeImage(idNum):
    imPath = os.path.join(imDir, str(idNum) + '.jpg')
    print("Removing "+imPath)
    os.remove(imPath)

def removeImagesWithoutMetadata():
    for imPath in glob.glob(os.path.join(imDir, '*.jpg')):
        idNum = re.findall('[0-9]+', imPath)
        idNum = int(idNum[0])
        if not idNum in releaseIds:
            removeImage(idNum)

def writeTextTrainingData():
    with open('releases.jsonl', 'w') as f:
        for idNum in tqdm(releaseIds[:300000]):
            try:
                release = database.getRelease(idNum)
                releaseStr = getTrainingStr(release)
                language = langdetect.detect(releaseStr)
                if language != 'en':
                    continue
                trainingStr = getTrainingStr(release)
                trainingJson = {'note': '### '+trainingStr}
                f.write(json.dumps(trainingJson) + '\n')
            except Exception as e:
                print(e)
                print(idNum)

def writeImageMetadataFile():
    with open(metadataFilePath, 'w') as metadataFile:
        for idNum in releaseIds:
            imPath = os.path.join(imDir, str(idNum) + '.jpg')
            if not os.path.exists(imPath):
                continue
            release = database.getRelease(idNum)

            if not 'genres' in release or release['year'] == 0:
                removeImage(idNum)
                continue

            try:
                promptStr = getPromptStr(release)
                metadata = {'file_name': os.path.split(imPath)[-1], 'text': promptStr}
                metadataFile.write(json.dumps(metadata) + '\n')
            except Exception as e:
                print(e)
                print(release)
                removeImage(idNum)

if __name__ == '__main__':
    random.seed(69)
    releaseIds = database.getAllReleaseIds()
    print('{} entries'.format(len(releaseIds)))
    random.shuffle(releaseIds)

    writeTextTrainingData()
    #writeMetadataFile()
    #removeImagesWithoutMetadata()