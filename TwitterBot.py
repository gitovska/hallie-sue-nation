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
            resource_owner_secret=os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
        )
        try:
            self.__mentions_df = pd.read_json('data/mentions.json')
        except ValueError and FileNotFoundError:
            self.__mentions_df = pd.DataFrame()


    # Private Class Methods

    def _get_request(self, url: str, params: dict = None) -> requests.Response:
        if params:
            response = self.__oauth.get(url, params=params)
        else:
            response = self.__oauth.get(url)

        if response.status_code != 200:
            raise Exception(f"Request returned an error: {response.status_code} {response.text}")
        return response

    def _post_request(self, tweet: str) -> requests.Response:
        """
        Sends a post request with a tweet in JSON format.
        Example: tweet = {"text": "Hello world!"}
        """
        response = self.__oauth.post("https://api.twitter.com/2/tweets", json=tweet)
        if response.status_code != 201:
            raise Exception(f"Request returned an error: {response.status_code} {response.text}")
        else:
            print("Tweet Posted!\n", json.dumps(response.json(), indent=4, sort_keys=True))
        return response

    def _get_user_id(self, username: str) -> str:
        base_url = "https://api.twitter.com/2/users"
        url = f"{base_url}/by?usernames={username}"
        params = {"user.fields": "id,description,created_at"}

        response = self._get_request(url, params=params)
        json_response = response.json()
        return json_response["data"][0]["id"]

    def _get_mention_ids(self, user_id: str) -> list[str]:
        base_url = "https://api.twitter.com/2/users"
        url = f"{base_url}/{user_id}/mentions"

        mentions_response = self._get_request(url)
        mentions = mentions_response.json()
        return [mention['id'] for mention in mentions['data']]

    def _query_mentions(self, mention_ids: list[str]) -> pd.DataFrame:

        if not self.__mentions_df.empty:
            query_ids = [tweet_id for tweet_id in mention_ids if tweet_id not in self.__mentions_df['id']]
        else:
            query_ids = mention_ids

        url = "https://api.twitter.com/2/tweets?"
        params = {"ids": ",".join(query_ids),
                  "tweet.fields": "created_at",
                  "expansions": "attachments.media_keys",
                  "media.fields": "media_key,alt_text,height,width,type,url",
                  }

        tweets_response = self._get_request(url, params)
        tweets_response_json = tweets_response.json()

        new_mentions_df = pd.DataFrame(columns=['id', 'text', 'media_key', 'created_at', 'processed'])
        for tweet in tweets_response_json['data']:
            tweet.setdefault('attachments', False)
            new_tweet = {}
            new_tweet['id'] = tweet['id']
            new_tweet['text'] = tweet['text']
            if tweet['attachments']:
                new_tweet['media_key'] = tweet['attachments']['media_keys'][0]
            else:
                new_tweet['media_key'] = None
            new_tweet['created_at'] = tweet['created_at']
            new_tweet['processed'] = False
            new_mentions_df.loc[len(new_mentions_df)] = new_tweet.values()
            new_mentions_df['id'].apply(lambda x: int(x))
            new_mentions_df['created_at'].apply(lambda x: pd.to_datetime(x))

        media_fields = ['media_key', 'height', 'width', 'url', 'type']
        media_df = pd.DataFrame(columns=media_fields)
        for tweet in tweets_response_json['includes']['media']:
            medium = {}
            for field in media_fields:
                medium[field] = tweet[field]
            media_df.loc[len(media_df)] = medium.values()

        media_df['height'].apply(lambda x: int(x))
        media_df['width'].apply(lambda x: int(x))

        return new_mentions_df.merge(media_df, how='outer', on='media_key')

    def _write_mentions(self, new_mentions: pd.DataFrame):
        full_mentions = pd.concat([self.__mentions_df, new_mentions], axis=0)
        full_mentions.reset_index(drop=True, inplace=True)
        if not os.path.isdir('data'):
            os.makedirs('data')
        full_mentions.to_json('data/mentions.json')


    # Public Class Methods

    def mentions(self, username: str):
        user_id = self._get_user_id(username)
        mention_ids = self._get_mention_ids(user_id)
        new_mentions = self._query_mentions(mention_ids)
        self._write_mentions(new_mentions)

    def tweet(self, tweet_string: str, image=None):
        if image:
            tweet = {"text": tweet_string}
        else:
            tweet = {"text": tweet_string}
        self._post_request(tweet)