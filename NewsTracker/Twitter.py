from NewsTracker import Configuration
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from NewsTracker.URLAnalyzer import URLAnalyzer
from NewsTracker.Google import GoogleSearch

import tweepy
import twint
from pandas import DataFrame
from typing import List
from enum import Enum
import json
from json import JSONEncoder, JSONDecoder


# HELPFUL TWINT REFERENCE:
# https://github.com/twintproject/twint/wiki/Configuration


class TwitterAnalyzer:


    class Tweet:

        class TweetEncoder(JSONEncoder):

            def default(self, obj):
                return obj.__dict__ 

        class TweetDecoder(JSONDecoder):

            def __init__(self, *args, **kwargs):
                json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

        class Origin(Enum):
            SEARCH = "search"
            REPLY = "reply"
            RETWEET = "retweet"
            QUOTE = "quote"

        class Stats:

            # WORK-IN-PROGRESS
            def __init__(self, like_count:          int,
                               reply_count:         int, 
                               retweet_count:       int,
                               quote_count:         int  = 0,
                               request_reply_count: bool = False,
                               request_quote_count: bool = True) -> None:

                # lacking data request
                # some data points cannot be provided by (all) APIs, so we approximate the value once we scraped for the according tweets
                self.request_reply_count = request_reply_count
                self.request_quote_count = request_quote_count

                # basic stats
                self.like_count    = like_count
                self.reply_count   = reply_count
                self.retweet_count = retweet_count
                self.quote_count   = quote_count

                # sentiment analysis
                # WIP

            @property
            def engagement_score(self) -> int:
                return (self.like_count + 2 * self.retweet_count + 3 * self.reply_count + 4 * self.quote_count) / 10

            def sentiment_analysis(self, text: str) -> None:
                ... # WIP

            @classmethod
            def from_dict(cls, stats_dict):
                stats = TweetStats(
                    like_count          = stats_dict["like_count"],
                    reply_count         = stats_dict["reply_count"],
                    retweet_count       = stats_dict["retweet_count"],
                    quote_count         = stats_dict["quote_count"],
                    request_reply_count = stats_dict["request_reply_count"],
                    request_quote_count = stats_dict["request_quote_count"]
                )
                # WIP: set sentiment analysis attributes
                return stats

            @classmethod
            def parse_twint(cls, tweet: DataFrame):
                stats = TwitterAnalyzer.Tweet.Stats(
                    like_count    = tweet["nlikes"],
                    reply_count   = tweet["nreplies"],
                    retweet_count = tweet["nretweets"]
                )
                stats.sentiment_analysis(tweet["tweet"])
                return stats

            @classmethod
            def parse_tweepy(cls, status):
                # find way to read reply count with twint instead
                stats = TwitterAnalyzer.Tweet.Stats(
                    like_count          = status.favorite_count,
                    reply_count         = 0,
                    retweet_count       = status.retweet_count,
                    request_reply_count = True
                )
                stats.sentiment_analysis(status.text)
                return stats

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
            self.stats = stats

            self.replies = []
            self.retweets = []
            self.quotes = []

            self._origin = origin.value

        @property
        def mention(self) -> str:
            return f"@{self.username}"

        @property
        def origin(self) -> Origin:
            return TweetOrigin(self._origin)

        def add_replies(self, tweets: List) -> None:
            self.replies += tweets
            if self.stats.request_reply_count:
                self.stats.replies = len(self.replies)

        def add_retweets(self, tweets: List) -> None:
            self.retweets += tweets

        def add_quotes(self, tweets: List) -> None:
            self.quotes += tweets
            if self.stats.request_quote_count:
                self.stats.quote_count = len(self.quotes)

        def __repr__(self) -> str:
            return f"Tweet\n\tID: {self.tweet_id}\n\tUsername: {self.username}\n\tText: {self.text}\n\tURL: {self.url}\n"

        @classmethod
        def from_dict(cls, tweet_dict):
            tweet = Tweet(
                tweet_id        = tweet_dict["tweet_id"],
                conversation_id = tweet_dict["conversation_id"],
                username        = tweet_dict["username"],
                date            = tweet_dict["date"],
                text            = tweet_dict["text"],
                url             = tweet_dict["url"],
                origin          = TweetOrigin(tweet_dict["_origin"]),
                stats           = TweetStats.from_dict(tweet_dict["stats"])
            )

            tweet.add_replies([Tweet.from_dict(reply) for reply in tweet_dict["replies"]])
            tweet.add_retweets([Tweet.from_dict(retweet) for retweet in tweet_dict["retweets"]])
            tweet.add_quotes([Tweet.from_dict(quote) for quote in tweet_dict["quotes"]])

            return tweet

        @classmethod
        def parse_twint(cls, tweet_data: DataFrame, origin: Origin):
            tweets = []
            for _, tweet_info in tweet_data.iterrows():
                stats = TweetStats.parse_twint(tweet_info)
                tweet = Tweet(
                    tweet_id        = tweet_info["id"],
                    conversation_id = tweet_info["conversation_id"],
                    username        = tweet_info["username"],
                    date            = tweet_info["date"].split()[0],
                    text            = tweet_info["tweet"],
                    url             = tweet_info["link"],
                    origin          = origin,
                    stats           = stats
                )
                tweets.append(tweet)
            return tweets

        @classmethod
        def parse_tweepy(cls, statuses, origin: Origin):
            tweets = []
            for status in statuses:
                stats = TweetStats.parse_tweepy(status)
                tweet = Tweet(
                    tweet_id        = status.id,
                    conversation_id = status.in_reply_to_status_id,
                    username        = status.user.screen_name,
                    date            = str(status.created_at).split()[0],
                    text            = status.text,
                    url             = f"https://twitter.com/twitter/statuses/{status.id}",
                    origin          = origin,
                    stats           = stats
                )
                tweets.append(tweet)
            return tweets


    def __init__(self, config: Configuration) -> None:
        self.config = config
        token = config.twitter_bearer_token

        self.auth   = tweepy.OAuth2BearerHandler(token)
        self.client = tweepy.Client(token)
        self.api    = tweepy.API(self.auth)
        self.sia = SIA() # vader sentiment analysis tool

        self.google = GoogleSearch(self.config, "twitter.com")

    def load_tweets_from_file(self, filename: str) -> List[Tweet]:
        with open(filename, "r") as file:
            json_list = json.load(file)
        tweets = [Tweet.from_dict(tweet) for tweet in json_list]
        print(tweets)

    def store_tweets_for_url(self, url: str, filename: str, limit: int = 100, recusion_depth = 1) -> None:
        tweets = self.get_tweets_for_url(url, limit, recusion_depth)
        with open(filename, "w") as file:
            json.dump(tweets, file, cls=TweetEncoder)

    def get_tweets_for_url(self, url: str, limit: int = 100, recusion_depth = 1) -> List[Tweet]:
        parent_tweets = self.search(url, limit=limit)
        return parent_tweets

        related_tweets = []
        for tweet in parent_tweets:
            related_tweets += self.get_related_tweets(tweet, limit, recusion_depth)
        related_tweets += parent_tweets
        print(len(related_tweets))

        return parent_tweets

        # url_analyser = URLAnalyzer(url)
        # title = url_analyser.title
        # google_search = self.google.search(title, num=10)
        # for result in google_search:
        #     print(f"{result.title}: {result.url}")

    def new_twint_config(self, hide_output: bool = True) -> twint.Config:
        c = twint.Config()
        c.Pandas= True
        c.Hide_output = hide_output
        return c

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

    def get_quotes(self, tweet: Tweet, limit: int = 10) -> List[Tweet]:
        response = self.client.get_quote_tweets(tweet.tweet_id, max_results=limit)
        if not response.data: return []

        statuses = [self.api.get_status(tweet.id) for tweet in response.data]
        return Tweet.parse_tweepy(statuses, TweetOrigin.QUOTE)

    def get_related_tweets(self, tweet: Tweet, limit: int = 1000, recursion_depth: int = 0, ignore_ids: List[str] = []) -> List[Tweet]:

        def filter_tweets(tweets: List[Tweet]):
            return [tweet for tweet in tweets if tweet.tweet_id not in ignore_ids]

        replies  = filter_tweets(self.get_replies(tweet, limit))
        retweets = filter_tweets(self.get_retweets(tweet))
        quotes   = filter_tweets(self.get_quotes(tweet, limit))

        tweet.add_replies(replies)
        tweet.add_retweets(retweets)
        tweet.add_quotes(quotes)

        tweets = replies + retweets + quotes
        ignore_ids += [tweet.tweet_id for tweet in tweets]

        results = []
        if recursion_depth > 0:
            for recursive_tweet in tweets:
                results += self.get_related_tweets(recursive_tweet, limit, recursion_depth - 1, ignore_ids)

        return tweets + results


Tweet = TwitterAnalyzer.Tweet
TweetOrigin = Tweet.Origin
TweetStats = Tweet.Stats
TweetEncoder = Tweet.TweetEncoder