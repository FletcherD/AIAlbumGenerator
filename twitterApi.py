import tweepy
import os
from dotenv import load_dotenv

load_dotenv()

apiKey = os.getenv('TWITTER_API_KEY')
apiSecret = os.getenv('TWITTER_API_SECRET')
accessToken = os.getenv('TWITTER_ACCESS_TOKEN')
accessTokenSecret = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
bearerToken = os.getenv('TWITTER_BEARER_TOKEN')

def createTweet(text, mediaPath):
    # Authenticate to Twitter
    auth = tweepy.OAuthHandler(apiKey, apiSecret)
    auth.set_access_token(accessToken, accessTokenSecret)
    api = tweepy.API(auth)
    client = tweepy.Client(bearerToken, apiKey, apiSecret, accessToken, accessTokenSecret)
    client.get_me()

    mediaFile = open(mediaPath, 'rb')
    media = api.media_upload(filename=mediaPath, file=mediaFile)

    client.create_tweet(text=text, media_ids=[media.media_id_string])
