import transformers
import time
import langdetect
import subprocess
import os

import discogsApi
import twitterApi

from inferAlbumDescription import inferAlbumDescription
from inferAlbumArtwork import inferAlbumArtwork

modelPath = './finetuned/'
eosToken = '<|endoftext|>'

def parseAlbumDescription(text):
    text = text.strip(eosToken)
    text = text.strip()
    lines = text.split('\n')
    def getField(line, expectedName):
        fieldName, fieldValue = line.split(': ', maxsplit=1)
        if fieldName != expectedName:
            print('bad field: '+expectedName)
            raise Exception
        return fieldValue

    release = {}
    release['artist'] = getField(lines[0], 'Artist').strip()
    release['title']  = getField(lines[1], 'Title').strip()
    release['genre']  = getField(lines[2], 'Genre').strip()
    release['year']   = getField(lines[3], 'Year').strip()
    release['tracklist'] = []
    release['tracklistStr'] = ''
    for l in lines[4:]:
        try:
            release['tracklist'].append(l.split(':')[1].strip())
            release['tracklistStr'] += '- ' + l.strip() + '\n'
        except:
            continue
    if len(release['tracklist']) == 0:
        raise Exception

    return release

def getTweetText(release):
    s = '{} - {}\n'.format(release['artist'], release['title'])
    s += 'Genre: {}\n'.format(release['genre'])
    s += 'Year: {}\n\n'.format(release['year'])
    s += release['tracklistStr']
    return s

def getAlbumImagePrompt(release):
    prompt = """{genre} album by {artist} titled '{title}' from {year}, tracks include {firstTracksStr}"""

    firstTracksStr = ', '.join(release['tracklist'][:3])
    return prompt.format(**release, firstTracksStr=firstTracksStr)

def trimImage(imagePath):
    subprocess.run(['bash', './trimImage.sh', imagePath])

def generate():
    timestamp = int(time.time())

    tokenizer = transformers.AutoTokenizer.from_pretrained(modelPath)
    generator = transformers.pipeline('text-generation', model=modelPath, tokenizer=tokenizer, pad_token_id=tokenizer.eos_token_id)
    transformers.set_seed(timestamp)

    release = None
    while release is None:
        releaseDescription = inferAlbumDescription(tokenizer, generator, temperature=1.3)
        try:
            release = parseAlbumDescription(releaseDescription)
        except:
            continue

        language = langdetect.detect(releaseDescription)
        print("Detected language: {}".format(language))
        if language != 'en':
            continue

        artistPopularity = discogsApi.getArtistPopularity(release['artist'])
        if artistPopularity > 0:
            print("Artist exists on Discogs - Popularity: {}".format(artistPopularity))
            if artistPopularity > 1000:
                continue
        else:
            print('--- Valid release, using ---')

    imagePromptStr = getAlbumImagePrompt(release)
    print(imagePromptStr)

    image = inferAlbumArtwork(imagePromptStr)

    imagePath = os.path.join('generated_images', f"{timestamp}.png")
    image.save(imagePath)

    trimImage(imagePath)

    tweetText = getTweetText(release)
    print(tweetText)
    while len(tweetText) > 280:
        tweetText = tweetText.split('\n')
        tweetText = '\n'.join(tweetText[:-1])
    twitterApi.createTweet(tweetText, imagePath)

if __name__ == '__main__':
    generate()
