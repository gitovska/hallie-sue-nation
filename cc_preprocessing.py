import re, openai, random
from dotenv import dotenv_values
from nltk import sent_tokenize


class Preprocessor():
    def __init__(self):
        self.__config = dotenv_values(".env")
        # load the api key for gpt3
        openai.api_key = self.__config['OPENAI_API_KEY']

    def _tweet_process(self, tweet: str) -> str:
        """

        :param tweet: original tweet string
        :return: the tweet after the word "dream", and all the hashtags in the original tweet

        motivation: You can not describe a dream in a tweet without using the word "dream".
        What comes after the word "dream" is then the content of the dream

        """
        # preprocess the string
        tweet = tweet.lower()
        content_part = tweet.split("dream",1)[1]

        # skip half word like t from dreamt
        content_part = content_part[2:]

        # capture the hashtags
        hashtag_pattern = "(#\w+)"
        res = re.finditer(hashtag_pattern,content_part)
        hashtags = []
        if res:
            for hashtag in res:
                hashtags.append(hashtag.group(0))
                #remove the hashtag from the original string
                content_part= content_part.replace(hashtag.group(0),"")

        # split url from the content
        content_part = re.split("http:",content_part)[0]

        return content_part

    def _get_noun_verb(self, tagger: object, tweet: str) -> tuple[set, set]:
        """

        :param tagger: the tagger used to tag the sentences
        :param tweet: the preprocessed tweet text that is going to be tagged
        :return: a set of nouns and a set of verbs in the tweet

        in this version: this function is not in use

        motivation:
        we want to capture the content of the dream, but also reconstruct the sentences about the dream into the format
        of a prompt. Therefore, we retrieve all the nouns and verbs (linguistically carry the most information) from the
        sentences, and construct them later into prompt with help of gpt3

        note: verbs like have, do have less interesting information about the dream content, so for the simplicity of the
        prompt, we remove them from the set
        """
        # first we tag the tweet text
        doc = tagger(tweet)

        # then we retrieve the nouns and verbs
        noun_list = []
        verb_list = []
        for token in doc:
            if token.pos_ == "NOUN":
                noun_list.append(token.lemma_)
            if token.pos_ == "VERB":
                verb_list.append(token.lemma_)

        # remove the word "night" from noun list, as it comes from "last night", which doesn't have information about
        # the dream itself
        noun_list = [ x for x in noun_list if x != "night"]

        # move verbs like "have", "do", that have less information about the dream too
        verb_list = [x for x in verb_list if x not in ["have","do"]]

        return set(noun_list),set(verb_list)

    def _generate_prompts(self, content_part: str) -> [str]:
        """
        :param content_part: tweet text
        :return: list of prompts from gpt
        1.let gpt3 continue the story.
        2. split the complex long sentences from gpt3 generated text into small simple text
        """

        # result prompt later for return
        prompts_list_from_pgt3 = []
        # prompt that goes into gpt3
        second_prompt = "Continue the story:" + content_part

        # send the prompt for gpt3 into gpt3 api
        completions = openai.Completion.create(
            engine="text-curie-001",
            prompt=second_prompt,
            max_tokens=300,
            n=1,
            temperature=1,  # high temperature gives gpt3 more creativity room
            presence_penalty=0.8,
        )

        # split the story
        story_text = completions.choices[0].text
        story_text.lower()

        # split the complex long sentences into shorter sentences
        to_replace = [".", "!", ",", "and", "then", "until","but","so","if","when","that","because"]
        for x in to_replace:
            story_text = story_text.replace(x, ".")
        prompts = sent_tokenize(story_text)

        # gives back first 8 prompts that passes the min length threshold
        prompts = [sent for sent in prompts if len(sent.split()) > 4]
        prompts = prompts[:8]

        # if there is not enough prompts
        if len(prompts) < 8:
            current_prompt_num = len(prompts)
            missing_num = 8 - len(prompts)
            for n in range(missing_num):
                # get a random number
                random_ix = random.randint(0,current_prompt_num-1)
                # append a random prompt from the prompt list to the list
                prompts.append(prompts[random_ix])

        # add generated prompts from gpt3 to list
        prompts_list_from_pgt3 = prompts_list_from_pgt3 + prompts

        # get rid of \n in the prompt
        prompts_list_from_pgt3 = [x.replace("\n","") for x in prompts_list_from_pgt3]
        prompts_list_from_pgt3 = [x.replace(".", "") for x in prompts_list_from_pgt3]
        prompts_list_from_pgt3 = [x.replace(",", "") for x in prompts_list_from_pgt3]

        # append style
        prompts_list_from_pgt3 = [x + " in the surreal style" for x in prompts_list_from_pgt3]

        return prompts_list_from_pgt3

    # public function
    def get_prompts(self, raw_tweet: str) -> list[str]:
        """

        :param raw_tweet: raw tweet text
        :return: a list of prompts
        """

        # preprocess raw_tweet
        tweet= self._tweet_process(raw_tweet)

        # generate prompts with gpt3
        prompt_list = self._generate_prompts(tweet)

        return prompt_list









