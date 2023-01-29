"""
1. Check for new mentions.
2. Write mentions to json.
3. Send unprocessed tweets to Prompt Generator.
4. Send first unprocessed tweet to Dreamer.
5. Send generated images to Style Transferer.
6. Tweet images.
7. Mark tweet as processed.
Repeat
"""

from TwitterBot import TwitterBot
from Preprocessor import Preprocessor
from Dreamer import Dreamer
import pandas as pd
import os
import subprocess, shlex
from dotenv import load_dotenv
import datetime


def get_unprocessed_tweets(df: pd.DataFrame) -> pd.DataFrame:
    unprocessed_tweets = df.where((df['processed'] == False) & (~df['media_key'].isnull()))
    unprocessed_tweets.dropna(inplace=True)
    unprocessed_tweets.sort_values(by=['created_at'], inplace=True)
    return unprocessed_tweets


def get_prompts(tweet: str) -> list[str]:
    prep = Preprocessor()
    return prep.get_prompts(tweet)


if __name__ == "__main__":

    # get mentions
    bot = TwitterBot()
    bot_name = "HallieSueNation"
    print(f"Getting mentions for {bot_name}")
    bot.mentions(bot_name)

    # get next unprocessed tweet
    df = pd.read_json('data/mentions.json',
                      dtype={"id": str, "text": str, "media_key": str, "created_at": datetime,
                             "height": int, "width": int, "url": str, "type": str})
    unprocessed_tweets = get_unprocessed_tweets(df)
    for _, tweet in unprocessed_tweets.iterrows():
        tweet_text = tweet['text']
        tweet_prompts = get_prompts(tweet_text)
        tweet_url = tweet['url']
        tweet_id = tweet['id']

        # dream
        print(f"Dreaming with tweet: '{tweet_text}' and the following prompts:")
        for prompt in tweet_prompts:
            print(f"\t{prompt}")
        #dreamer = Dreamer()
        #dreamer.dream(next_tweet_prompts, tweet_id=next_tweet_id, image_url=next_tweet_url)

        # mark tweet as processed
        print(f"Marking tweet {tweet_id} as processed")
        df.loc[df['id'] == tweet['id'], 'processed'] = True
        os.remove('data/mentions.json')
        df.to_json('data/mentions.json')

        # tweet back with dream sequence
        # Note: multiple methods have been attempted to tweet back the dream sequence to the user.
        # Attempt: tweet reply. Issue: Twitter refuses our requests with 403 forbidden error.
        # Attempt: sending the dream sequence to a privately hosted webserver running NGINX in a docker container,
        # and providing a link to the image in the tweet. Issue: A connection can be established between the two servers,
        # and the files transferred manually, but this fails within a script, be it here in main, a shell script, or server side
        # sync tools like lsync.

        # ssh syncing of dreams to webserver

        # load_dotenv(".env")
        # dream_transfer = shlex.split(
        #     f"./dream-transfer.sh {os.getenv('PORT')} {os.getenv('USER')} {os.getenv('DOMAIN')}")
        # result = subprocess.run(dream_transfer, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # print(result.stdout)
        # print(result.stderr)

        # bot.tweet(f"@{bot.username_lookup(next_tweet_id)}, you dream sequence is at halliesuenation.ad.rienne.de/{next_tweet_id}_dream_grid.bmp")
        print(f"Tweet: @{bot.username_lookup(tweet_id)}, look at your dream")

    print("All tweets processed. I am having a 10 minute nap. Maybe I'll have a dream of my own")
