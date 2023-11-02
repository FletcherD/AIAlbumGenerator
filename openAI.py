import openai
import requests
import time

secretKey = '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'

openai.api_key = secretKey

prompt = """Album artwork for the album "{title}" by {artist}, released in {year}, in the genres {genre}"""

def generateAlbumArtwork(release):
    thisPrompt = prompt.format(**release)
    print(thisPrompt)

    response = openai.Image.create(
        prompt=thisPrompt,
        n=1,
        size="1024x1024"
    )
    image_url = response['data'][0]['url']

    r = requests.get(image_url)
    imagePath = '{}.png'.format(int(time.time()))
    with open(imagePath, 'wb') as f:
        f.write(r.content)

    return imagePath
