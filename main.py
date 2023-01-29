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


def get_next_tweet(df: pd.DataFrame) -> pd.DataFrame:
    unprocessed_tweets = df.where((df['processed'] == False) & (~df['media_key'].isnull()))
    unprocessed_tweets.sort_values(by=['created_at'], inplace=True)
    return unprocessed_tweets.head(1)

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
    df = pd.read_json('data/mentions.json')
    next_tweet = get_next_tweet(df)
    next_tweet_text = next_tweet['text'].values[0]
    next_tweet_prompts = get_prompts(next_tweet_text)
    next_tweet_url = next_tweet['url'].values[0]
    next_tweet_id = (int(next_tweet['id'].values[0]))

    # dream
    print(f"Dreaming with tweet: '{next_tweet_text}' and the following prompts:")
    for prompt in next_tweet_prompts:
        print(f"\t{prompt}")
    dreamer = Dreamer()
    dreamer.dream(prompts, tweet_id=next_tweet_id, image_url=next_tweet_url)

    # tweet back with dream sequence
    # bot.reply(next_tweet_id, image)
