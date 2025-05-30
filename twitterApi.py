import tweepy

apiKey = 'gRKckPDC7xaXpuhMlyCBeC0Rx'
apiSecret = 'dxws6ygMXNK6Zdto80IU3cmod2xDCFpSybEeYYjuhj4I5ICC5r'
accessToken = '1718853436496891904-i8F8gRAEYPAPmwHHitAWwg4DFeyN0P'
accessTokenSecret = 'B29QbgWxYyuBrD4yYUfWV2av7mYMVp0Fanqf9Bgc3EN1X'
bearerToken = 'AAAAAAAAAAAAAAAAAAAAAAERqwEAAAAAOKo4XozFFiwjfBSp%2FPnAMPftbBI%3Dp4U9UtN7dNtMmhwvZdcS9cJgCtSkvStuR3lyt5JCUVaetdTYOM'

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
