from NewsTracker import Configuration
from NewsTracker.URLAnalyzer import URLAnalyzer
from NewsTracker.Google import GoogleSearch

import tweepy
import twint
from pandas import DataFrame
from typing import List
from enum import Enum


class TwitterAnalyzer:

    class Tweet:

        class Origin(Enum):
            SEARCH = 0
            REPLY = 1
            RETWEET = 2
            QUOTE = 3

        class Stats:

            # WORK-IN-PROGRESS
            def __init__(self) -> None:
                pass

            @classmethod
            def parse_twint(cls, tweet: DataFrame):
                return TwitterAnalyzer.Tweet.Stats()

            @classmethod
            def parse_tweepy(cls, status):
                return TwitterAnalyzer.Tweet.Stats()

        def __init__(self, tweet_id: str, 
                           conversation_id: str, 
                           username: str, 
                           date,
                           text: str, 
                           url: str,
                           origin: Origin,
                           stats: Stats) -> None:

            self.tweet_id = tweet_id
            self.conversation_id = conversation_id
            self.username = username
            self.date = date
            self.text = text
            self.url = url
            self.origin = origin
            self.stats = stats

        @property
        def mention(self) -> str:
            return f"@{self.username}"

        @classmethod
        def parse_twint(cls, tweet_data: DataFrame, origin: Origin):
            tweets = []
            for _, tweet_info in tweet_data.iterrows():
                stats = TweetStats.parse_twint(tweet_info)
                tweet = Tweet(tweet_id        = tweet_info["id"],
                              conversation_id = tweet_info["conversation_id"],
                              username        = tweet_info["username"],
                              date            = tweet_info["date"].split()[0],
                              text            = tweet_info["tweet"],
                              url             = tweet_info["link"],
                              origin          = origin,
                              stats           = stats)
                tweets.append(tweet)
            return tweets

        @classmethod
        def parse_tweepy(cls, statuses, origin: Origin):
            tweets = []
            for status in statuses:
                stats = TweetStats.parse_tweepy(status)
                tweet = Tweet(tweet_id        = status.id,
                              conversation_id = status.in_reply_to_status_id,
                              username        = status.user.screen_name,
                              date            = str(status.created_at).split()[0],
                              text            = status.text,
                              url             = f"https://twitter.com/twitter/statuses/{status.id}",
                              origin          = origin,
                              stats           = stats)
                tweets.append(tweet)
            return tweets

        def __repr__(self) -> str:
            return f"Tweet\n\tID: {self.tweet_id}\n\tUsername: {self.username}\n\tText: {self.text}\n\tURL: {self.url}\n"

    def __init__(self, config: Configuration) -> None:
        self.config = config
        token = config.twitter_bearer_token

        self.auth   = tweepy.OAuth2BearerHandler(token)
        self.client = tweepy.Client(token)
        self.api    = tweepy.API(self.auth)

        self.google = GoogleSearch(self.config, "twitter.com")

    def new_twint_config(self, hide_output: bool = True) -> twint.Config:
        c = twint.Config()
        c.Pandas= True
        c.Hide_output = hide_output
        return c

    def analyze_url(self, url: str) -> None:
        # url_analyser = URLAnalyzer(url)
        # title = url_analyser.title

        twitter_search = self.search(url, limit=10)

        related_tweets = []
        for tweet in twitter_search:
            related_tweets += self.get_related_tweets(tweet)
        related_tweets += [twitter_search]

        print(related_tweets)
        print(len(related_tweets))

        # google_search = self.google.search(title, num=10)
        # for result in google_search:
        #     print(f"{result.title}: {result.url}")

    def search(self, search_term: str, limit: int = 10, hide_output: bool = True) -> List[Tweet]:
        c = self.new_twint_config(hide_output)
        c.Search = search_term
        c.Limit = limit
        twint.run.Search(c)

        results = twint.output.panda.Tweets_df
        return Tweet.parse_twint(results, TweetOrigin.SEARCH)

    def get_replies(self, tweet: Tweet, limit: int = 1000) -> List[Tweet]:
        c = self.new_twint_config()
        c.To = tweet.mention
        c.Since = tweet.date
        c.Limit = limit
        twint.run.Search(c)

        results = twint.output.panda.Tweets_df
        if results.empty: return []

        filtered_results = results[results["conversation_id"] == tweet.conversation_id]
        return Tweet.parse_twint(filtered_results, TweetOrigin.REPLY)

    def get_retweets(self, tweet: Tweet) -> List[Tweet]:
        results = self.api.get_retweets(tweet.tweet_id)
        return Tweet.parse_tweepy(results, TweetOrigin.RETWEET)

    # WORK-IN-PROGRESS
    def get_quotes(self, tweet: Tweet, limit: int = 10) -> List[Tweet]:
        results = self.client.get_quote_tweets(tweet.tweet_id, max_results=limit)
        return Tweet.parse_tweepy(results, TweetOrigin.QUOTE)

    def get_related_tweets(self, tweet: Tweet, limit: int = 1000, recursion_depth: int = 0) -> List[Tweet]:
        replies = self.get_replies(tweet, limit)
        retweets = self.get_retweets(tweet)
        quotes = [] # self.get_quotes(tweet)  # WORK-IN-PROGRESS

        tweets = replies + retweets + quotes

        results = []
        if recursion_depth > 0:
            for recursive_tweet in tweets:
                results += self.get_related_tweets(recursive_tweet, limit, recursion_depth - 1)

        return tweets + results


Tweet = TwitterAnalyzer.Tweet
TweetOrigin = Tweet.Origin
TweetStats = Tweet.Stats