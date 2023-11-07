import openai
import requests
import time
import os

secretKey = os.getenv('OPENAI_API_KEY')

openai.api_key = secretKey

prompt = """Album artwork for an album titled "{title}" by the artist {artist}, released in {year}, in the genres {genre}"""

def generateAlbumArtwork(release):
    thisPrompt = prompt.format(**release)
    print(thisPrompt)

    response = openai.Image.create(
        model="dall-e-3",
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
