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

#TODO: if artist == Various and tracks > 25
def processAlbumDescription(text):
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
        return None

    s = '{} - {}\n'.format(release['artist'], release['title'])
    s += 'Genre: {}\n'.format(release['genre'])
    s += 'Year: {}\n\n'.format(release['year'])
    s += release['tracklistStr']
    release['text'] = s

    print(release['text'])

    language = langdetect.detect(release['text'])
    print("Detected language: {}".format(language))
    if language != 'en':
        print("Not English language - skipping")
        return None

    artistPopularity = discogsApi.getArtistPopularity(release['artist'])
    if artistPopularity > 0:
        print("ARTIST EXISTS - Popularity: {}".format(artistPopularity))
        if artistPopularity > 1000:
            return None
        else:
            return release
    else:
        print('VALID')
        return release

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
        release = inferAlbumDescription(tokenizer, generator, temperature=1.3)
        release = processAlbumDescription(release)

    imagePromptStr = getAlbumImagePrompt(release)
    print(imagePromptStr)

    image = inferAlbumArtwork(imagePromptStr)

    imagePath = os.path.join('generated_images', f"{timestamp}.png")
    image.save(imagePath)

    trimImage(imagePath)

    releaseText = release['text']
    while len(releaseText) > 280:
        releaseText = releaseText.split('\n')
        releaseText = '\n'.join(releaseText[:-1])
    twitterApi.createTweet(releaseText, imagePath)

if __name__ == '__main__':
    generate()
