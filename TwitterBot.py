import requests
from requests_oauthlib import OAuth1Session
import os
import json
from dotenv import load_dotenv
import pandas as pd


# The base code of this twitter bot is a refactoring of twitterdev code from the following sources:
    # github.com/twitterdev/Twitter-API-v2-sample-code
        # Tweet-Lookup/get_tweets_with_bearer_token.py
        # User-Lookup/get_users_with_bearer_token.py
        # User-Mention-Timeline/user_mentions.py
        # Manage-Tweets/create_tweet.py
class TwitterBot:
    def __init__(self):
        load_dotenv(".env")
        self.__oauth = OAuth1Session(
            client_key=os.getenv('TWITTER_CONSUMER_KEY'),
            client_secret=os.getenv('TWITTER_CONSUMER_SECRET'),
            resource_owner_key=os.getenv('TWITTER_ACCESS_TOKEN'),
            resource_owner_secret=os.getenv('TWITTER_ACCESS_TOKEN_SECRET'),
        )
        try:
            self.__mentions_df = pd.read_json('data/mentions.json')
        except ValueError:
            self.__mentions_df = None

    def _get_request(self, url, params=None):
        if params:
            response = self.__oauth.get(url, params=params)
        else:
            response = self.__oauth.get(url)

        if response.status_code != 200:
            raise Exception(f"Request returned an error: {response.status_code} {response.text}")
        return response

    def _post_request(self, tweet):
        """
        Sends a post request with a tweet in JSON format.
        Example: tweet = {"text": "Hello world!"}
        """
        response = self.__oauth.post("https://api.twitter.com/2/tweets", json=tweet)
        if response.status_code != 201:
            raise Exception(f"Request returned an error: {response.status_code} {response.text}")

        # Saving the response as JSON and printing
        json_response = response.json()
        print(f"Status: {response.status_code} - Tweet Posted!\n", json.dumps(json_response, indent=4, sort_keys=True))

    def _get_user_id(self, username):
        base_url = "https://api.twitter.com/2/users"
        url = f"{base_url}/by?usernames={username}"
        params = {"user.fields": "id,description,created_at"}

        response = self._get_request(url, params=params)
        json_response = response.json()
        return json_response["data"][0]["id"]

    def _get_mention_ids(self, user_id):
        base_url = "https://api.twitter.com/2/users"
        url = f"{base_url}/{user_id}/mentions"

        mentions_response = self._get_request(url)
        mentions = mentions_response.json()
        return [mention['id'] for mention in mentions['data']]

    def _query_mentions(self, mention_ids):

        if self.__mentions_df:
            query_ids = [tweet_id for tweet_id in mention_ids if tweet_id not in self.__mentions_df['id']]
        else:
            query_ids = mention_ids

        url = "https://api.twitter.com/2/tweets?"
        params = {"ids": ",".join(query_ids),
                  "expansions": "attachments.media_keys",
                  "media.fields": "duration_ms,height,media_key,preview_image_url,public_metrics,type,url,width,alt_text",
                  }

        tweets_response = self._get_request(url, params)

        return tweets_response.json()

    def _write_mentions(self, mentions):
        pass

    def mentions(self, username):
        user_id = self._get_user_id(username)
        mention_ids = self._get_mention_ids(user_id)
        mentions = self._query_mentions(mention_ids)
        self._write_mentions(mentions)
        print(json.dumps(mentions))

    def tweet(self, tweet_string, image=None):
        if image:
            tweet = {"text": tweet_string}
        else:
            tweet = {"text": tweet_string}
        self._post_request(tweet)


bot = TwitterBot()
bot.mentions("HallieSueNation")
