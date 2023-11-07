import transformers
import time
import langdetect

import discogsApi
import openAI
import twitterApi

modelPath = './finetuned/'

eosToken = '<|endoftext|>'

#TODO: if artist == Various and tracks > 25
def processRelease(text):
    text = text.strip(eosToken)
    text = text.strip()
    lines = text.split('\n')
    def getField(line, expectedName):
        fieldName, fieldValue = line.split(': ', maxsplit=1)
        if fieldName != expectedName:
            print('bad field: '+expectedName)
            raise Exception
        return fieldValue

    try:
        release = {}
        release['artist'] = getField(lines[0], 'Artist')
        release['title']  = getField(lines[1], 'Title')
        release['genre']  = getField(lines[2], 'Genre')
        release['year']   = getField(lines[3], 'Year')
        release['tracklist'] = '\n'.join(['- '+l.strip() for l in lines[4:]])

        s = '{} - {}\n'.format(release['artist'], release['title'])
        s += 'Genre: {}\n'.format(release['genre'])
        s += 'Year: {}\n\n'.format(release['year'])
        s += release['tracklist']
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
    except Exception as e:
        print('INVALID')
        print(text)
        print(e)
        return None

def generate():
    tokenizer = transformers.AutoTokenizer.from_pretrained(modelPath)
    generator = transformers.pipeline('text-generation', model=modelPath, tokenizer=tokenizer)
    transformers.set_seed(int(time.time()))

    def generateRelease():
        result = generator(tokenizer.eos_token + '\nArtist:', max_length=1000, num_return_sequences=1)
        return result[0]["generated_text"]

    release = None
    while release is None:
        release = generateRelease()
        release = processRelease(release)
    imagePath = openAI.generateAlbumArtwork(release)

    releaseText = release['text']
    while len(releaseText) > 280:
        releaseText = releaseText.split('\n')
        releaseText = '\n'.join(releaseText[:-1])
    twitterApi.createTweet(releaseText, imagePath)

if __name__ == '__main__':
    generate()
