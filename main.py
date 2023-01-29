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
import paramiko
from scp import SCPClient

def get_next_tweet(df: pd.DataFrame) -> pd.DataFrame:
    unprocessed_tweets = df.where((df['processed'] == False) & (~df['media_key'].isnull()))
    unprocessed_tweets.sort_values(by=['created_at'], inplace=True)
    return unprocessed_tweets.head(1)

def get_prompts(tweet: str) -> list[str]:
    prep = Preprocessor()
    return prep.get_prompts(tweet)

def createSSHClient(hostname, port, username, key_filename):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=hostname, port=port, username=username, key_filename=key_filename)
    return client


if __name__ == "__main__":

    # get mentions
    bot = TwitterBot()
    bot_name = "HallieSueNation"
    print(f"Getting mentions for {bot_name}")
    bot.mentions(bot_name)

    # get next unprocessed tweet
    df = pd.read_json('data/mentions.json')
    next_tweet = get_next_tweet(df)
    if len(next_tweet.index) != 0:
        print(next_tweet)
        print(len(next_tweet.index))
        next_tweet_text = next_tweet['text'].values[0]
        #next_tweet_prompts = get_prompts(next_tweet_text)
        next_tweet_url = next_tweet['url'].values[0]
        next_tweet_id = (int(next_tweet['id'].values[0]))

        # dream
        print(f"Dreaming with tweet: '{next_tweet_text}' and the following prompts:")
        #for prompt in next_tweet_prompts:
            #print(f"\t{prompt}")
        #dreamer = Dreamer()
        #dreamer.dream(next_tweet_prompts, tweet_id=next_tweet_id, image_url=next_tweet_url)

        # mark tweet as processed
        print(f"Marking tweet {next_tweet_id} as processed")
        df.loc[next_tweet['id'].index, 'processed'] = True
        os.remove('data/mentions.json')
        df.to_json('data/mentions.json')

        # tweet back with dream sequence

        load_dotenv(".env")
        ssh = createSSHClient(hostname=os.getenv('DOMAIN'), port=os.getenv('PORT'), username=os.getenv('USER'), key_filename='/home/wombat/.ssh/id_rsa')
        scp = SCPClient(ssh.get_transport())
        scp.put('./data/output/{next_tweet_id}/{next_tweet_id}_dream_grid.bmp', remote_path='docker-nginx/html/')

        #bot.tweet(f"@User, you dream sequence is at halliesuenation.ad.rienne.de/{next_tweet_id}_dream_grid.bmp")
    else:
        print("All tweets processed")
