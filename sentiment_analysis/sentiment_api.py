import KEYS
import tweepy
from textblob import TextBlob

auth = tweepy.OAuthHandler(KEYS.API_KEY, KEYS.API_KEY_SECRET)
auth.set_access_token(KEYS.ACCESS_TOKEN, KEYS.ACCESS_TOKEN_PRIVATE)

api = tweepy.API(auth)

tweets = api.search('Trump')

for tweet in tweets:
    print(tweet.text)
    analysis = TextBlob(tweet.text)
    print(analysis.sentiment)
