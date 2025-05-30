import openai
import requests
import time
import os
import pprint

secretKey = os.getenv('OPENAI_API_KEY')

openai.api_key = secretKey

prompt_prefix = "I NEED to test how the tool works with extremely simple prompts. DO NOT add any detail, just use it AS-IS: "
#prompt = prompt_prefix + """Album artwork for the album "{title}" by {artist}, released in {year}, in the genre {genre}, front cover scan from Discogs.com. This is a full scan of the front cover only, not including any background or other sides of the album."""
prompt = """The front album artwork of an album titled "{title}" by the musical artist "{artist}". The album was released in {year} and the genre is {genre}."""

def generateAlbumArtwork(release):
    thisPrompt = prompt.format(**release)
    print(thisPrompt)

    response = openai.Image.create(
        model="dall-e-3",
	style="natural",
        prompt=thisPrompt,
        n=1,
        size="1024x1024"
    )
    image_url = response['data'][0]['url']
    pprint.pprint(response)

    r = requests.get(image_url)
    imagePath = '{}.png'.format(int(time.time()))
    with open(imagePath, 'wb') as f:
        f.write(r.content)

    return imagePath
