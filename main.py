from TwitterBot import TwitterBot
from Preprocessor import Preprocessor
from Dreamer import Dreamer
import pandas as pd
import os
import datetime
from random import sample


# The main of Hallie-Sue does the following:
#
# 1. Check Hallie-Sue's Twitter for new mentions.
# 2. For each unprocessed tweet:
#      - Send the text of unprocessed tweet to the Preprocessor.
#      - Send the image and eight prompts of each tweet to Dreamer.
#      - Tweet dream sequence grid.
#      - Mark tweet as processed and update mentions.json


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
        dreamer = Dreamer()
        dreamer.dream(tweet_prompts, tweet_id, tweet_url)

        # mark tweet as processed
        print(f"Marking tweet {tweet_id} as processed")
        df.loc[df['id'] == tweet['id'], 'processed'] = True
        os.remove('data/mentions.json')
        df.to_json('data/mentions.json')

        # tweet back with dream sequence
        replies = ["this is how your dream made me feel", "do you like it?", "I dreamt this up for you", "your dream made me think of this:"]
        reply = sample(replies, 1)[0]
        reply_tweet_text = f"@{bot.username_lookup(tweet_id)}, {reply}"
        bot.reply(tweet_id, f"./data/output/{tweet_id}/{tweet_id}_dream_grid.bmp")

    print("All tweets processed. I am having a 10 minute nap. Maybe I'll have a dream of my own")
