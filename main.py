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
import pandas

from TwitterBot import TwitterBot
from Preprocessor import Preprocessor
from Dreamer import Dreamer
import pandas as pd


def get_next_tweet(df: pd.DataFrame) -> pd.DataFrame:
    unprocessed_tweets = df.where(df['processed'] == False)
    unprocessed_tweets.sort_values(by=['created_at'], inplace=True, ascending=False)
    return unprocessed_tweets.head(1)

def get_prompts(tweet: str) -> list[str]:
    #prep = Preprocessor()
    #prompts = prep.get_prompts(prompt)
    return [tweet] * 8


if __name__ == "__main__":

    bot = TwitterBot()
    # bot.mentions("HallieSueNation")
    df = pd.read_json('data/mentions.json')
    next_tweet = get_next_tweet(df)
    prompts = get_prompts(next_tweet['text'].values[0])
    url = next_tweet['url'].values[0]
    next_tweet_id = (int(next_tweet['id'].values[0]))

    #dreamer = Dreamer()
    #dreamer.dream(prompts, tweet_id=next_tweet_id, image_url=url)

    #bot.reply(next_tweet_id, image)
