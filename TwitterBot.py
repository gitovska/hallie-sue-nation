import requests
from requests_oauthlib import OAuth1Session
import os
from dotenv import load_dotenv
import pandas as pd
import json
import datetime


# The base code of this twitter bot is a refactoring of twitterdev code from the following sources:
# github.com/twitterdev/Twitter-API-v2-sample-code
# Tweet-Lookup/get_tweets_with_bearer_token.py
# User-Lookup/get_users_with_bearer_token.py
# User-Mention-Timeline/user_mentions.py
# Manage-Tweets/create_tweet.py

class TwitterBot:
    """
    A Twitter Bot that can get a user's new mentions, save them to a json,
    and reply to the mentions with an image and tweet.
    """
    def __init__(self):
        load_dotenv(".env")
        self.__oauth = OAuth1Session(
            client_key=os.getenv('TWITTER_CONSUMER_KEY'),
            client_secret=os.getenv('TWITTER_CONSUMER_SECRET'),
            resource_owner_key=os.getenv('TWITTER_ACCESS_TOKEN'),
            resource_owner_secret=os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
        )
        try:
            self.__mentions_df = pd.read_json('data/mentions.json',
                                              dtype={"id": str, "text": str, "media_key": str, "created_at": datetime,
                                                     "height": int, "width": int, "url": str, "type": str})
        except ValueError and FileNotFoundError:
            self.__mentions_df = pd.DataFrame()
            self.__mentions_df.to_json('data/mentions.json')

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
        json_response = json.loads(response.text, parse_float=str(), parse_int=str())
        return json_response["data"][0]["id"]

    def _get_mention_ids(self, user_id: str) -> list[str]:
        base_url = "https://api.twitter.com/2/users"
        url = f"{base_url}/{user_id}/mentions"

        mentions_response = self._get_request(url)
        mentions = json.loads(mentions_response.text, parse_float=str(), parse_int=str())
        return [mention['id'] for mention in mentions['data']]

    def _query_mentions(self, mention_ids: list[str]):
        #-> pd.DataFrame | None
        if not self.__mentions_df.empty:
            id_list = list(self.__mentions_df['id'].values)
            query_ids = [tweet_id for tweet_id in mention_ids if tweet_id not in id_list]
        else:
            query_ids = mention_ids

        if query_ids:
            url = "https://api.twitter.com/2/tweets?"
            params = {"ids": ",".join(query_ids),
                      "tweet.fields": "created_at",
                      "expansions": "attachments.media_keys",
                      "media.fields": "media_key,alt_text,height,width,type,url",
                      }

            tweets_response = self._get_request(url, params)
            tweets_response_json = json.loads(tweets_response.text, parse_float=str(), parse_int=str())

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

            merge_df = new_mentions_df.merge(media_df, how='outer', on='media_key')
            return merge_df
        else:
            return None

    def _write_new_mentions(self, new_mentions: pd.DataFrame):
        if not os.path.isdir('data'):
            os.makedirs('data')
        new_mentions.to_json('data/mentions.json')

    def _upload_image(self, image: str) -> str:
        """

        :param image: path to the image
        :return: media_id as string
        this function retrieves the id of the images, which is needed for upload image later
        """
        url = "https://upload.twitter.com/1.1/media/upload.json"
        files = {"media": open(image, 'rb')}

        response = self.__oauth.post(url, files=files)

        if response.status_code in [200, 201]:
            media_id = json.loads(response.text, parse_float=str(), parse_int=str())['media_id']
            return media_id
        else:
            return False

    # Public Class Methods

    def mentions(self, username: str):
        user_id = self._get_user_id(username)
        mention_ids = self._get_mention_ids(user_id)
        new_mentions = self._query_mentions(mention_ids)
        if type(new_mentions) != type(None):
            self._write_new_mentions(new_mentions)

    def tweet(self, tweet_string: str, image=None):
        if image:
            tweet = {"text": tweet_string}
        else:
            tweet = {"text": tweet_string}
        self._post_request(tweet)

    def username_lookup(self, tweet_id):
        tweet_id = tweet_id
        url = f'https://api.twitter.com/2/tweets?ids={tweet_id}'

        params = {"expansions": "author_id"}
        response = self.__oauth.get(url, params=params)
        if response.status_code in [200, 201]:
            username = json.loads(response.text, parse_float=str(), parse_int=str())
            return username["includes"]["users"][0]["username"]
        else:
            print("Failed to get username")
            print(response.status_code)
            return False

    def refresh(self):
        refresh_url = 'https://api.twitter.com/2/oauth2/token'
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        params = {"refresh_token": os.getenv('TWITTER_ACCESS_TOKEN'),
                  "grant_type": "refresh_token",
                  "client_id": os.getenv('TWITTER_CONSUMER_KEY')}

        response = requests.post(url=refresh_url, headers=headers, params=params)
        print("refreshing")
        print(response)

    def reply(self, tweet_id: str, image: str):
        """

        :param tweet_id: the tweet_id as str
        :param image: the path to the image
        :return:
        """

        username = self.username_lookup(tweet_id)
        media_id = self._upload_image(image)
        params = {'text': f"@{username} This is how your dream makes me feel.",
                  'media': {"media_ids": [str(media_id)]},
                  "reply": {"in_reply_to_tweet_id": str(tweet_id)}}
        print(params)
        url = "https://api.twitter.com/2/tweets"
        response = self.__oauth.post(url, json=params)

        if response.status_code in [200, 201]:
            print("Tweet was successfully posted")
        else:
            print("Something went wrong")
            print(response.status_code)
            print(response)
