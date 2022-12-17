from NewsTracker import Configuration
from NewsTracker.URLAnalyzer import URLAnalyzer
from NewsTracker.Google import GoogleSearch

import tweepy
import twint
import json
import sys

from typing import List, Tuple
from enum import Enum
from json import JSONEncoder
from datetime import datetime, timedelta
from io import StringIO 

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pandas import DataFrame
from tqdm import tqdm


# HELPFUL TWINT REFERENCE:
# https://github.com/twintproject/twint/wiki/Configuration


class TwintCapturing(list):

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout

    @classmethod
    def search(cls, config: twint.Config):
        config.Pandas = True

        with cls() as _:
            try:
                twint.run.Search(config)
            except:
                return DataFrame()

        return twint.output.panda.Tweets_df


class TwitterAnalyzer:


    class Tweet:

        DATAFRAME_COLUMS = ["Tweet ID", 
                            "Conversation ID", 
                            "Username", 
                            "Date", 
                            "Text", 
                            "URL",
                            "Origin", 
                            "Likes", 
                            "Replies", 
                            "Retweets", 
                            "Quotes",
                            "Negative Sentiment Score",
                            "Neutral Sentiment Score",
                            "Positive Sentiment Score",
                            "Compound Sentiment Score"]

        class TweetEncoder(JSONEncoder):
            def default(self, obj):
                return obj.__dict__ 

        class Origin(Enum):
            SEARCH = "search"
            REPLY = "reply"
            RETWEET = "retweet"
            QUOTE = "quote"

        class Stats:

            SIA = SentimentIntensityAnalyzer() # vader sentiment analysis tool

            # WORK-IN-PROGRESS
            def __init__(self, like_count:          int,
                               reply_count:         int, 
                               retweet_count:       int,
                               quote_count:         int  = 0,
                               negative_sentiment:  bool = None,
                               neutral_sentiment:   bool = None,
                               positive_sentiment:  bool = None,
                               coupound_sentiment:  bool = None,
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
                self.negative_sentiment = negative_sentiment
                self.neutral_sentiment  = neutral_sentiment
                self.positive_sentiment = positive_sentiment
                self.coupound_sentiment = coupound_sentiment

            @property
            def engagement_score(self) -> int:
                return (self.like_count + 2 * self.retweet_count + 3 * self.reply_count + 4 * self.quote_count) / 10

            @property
            def is_positive(self) -> bool:
                if self.coupound_sentiment:
                    return self.coupound_sentiment > 0
                return None

            def sentiment_analysis(self, text: str) -> None:
                polarity = TweetStats.SIA.polarity_scores(text)
                self.negative_sentiment = polarity["neg"]
                self.neutral_sentiment = polarity["neu"]
                self.positive_sentiment = polarity["pos"]
                self.coupound_sentiment = polarity["compound"]

            @classmethod
            def from_dict(cls, stats_dict):
                return TweetStats(
                    like_count          = stats_dict["like_count"],
                    reply_count         = stats_dict["reply_count"],
                    retweet_count       = stats_dict["retweet_count"],
                    quote_count         = stats_dict["quote_count"],
                    negative_sentiment  = stats_dict["negative_sentiment"],
                    neutral_sentiment   = stats_dict["neutral_sentiment"],
                    positive_sentiment  = stats_dict["positive_sentiment"],
                    coupound_sentiment  = stats_dict["coupound_sentiment"],
                    request_reply_count = stats_dict["request_reply_count"],
                    request_quote_count = stats_dict["request_quote_count"]
                )

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
            self.date = str(datetime.strptime(date, '%Y-%m-%d %H:%M:%S'))
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

        @property
        def search_date(self) -> str:
            return str(datetime.strptime(self.date, '%Y-%m-%d %H:%M:%S') - timedelta(days=1)).split()[0]

        @property
        def childern(self) -> List:
            return self.replies + self.retweets + self.quotes

        @property
        def dataframe_row(self) -> List:
            return [
                self.tweet_id,
                self.conversation_id,
                self.username,
                self.date,
                self.text,
                self.url,
                self._origin,
                self.stats.like_count,
                self.stats.reply_count,
                self.stats.retweet_count,
                self.stats.quote_count,
                self.stats.negative_sentiment,
                self.stats.neutral_sentiment,
                self.stats.positive_sentiment,
                self.stats.coupound_sentiment
            ]

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
                    date            = tweet_info["date"],
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
                    date            = str(status.created_at).split("+")[0],
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
        self.client = tweepy.Client(token, wait_on_rate_limit=True)
        self.api    = tweepy.API(self.auth, wait_on_rate_limit=True)

        self.google = GoogleSearch(self.config, "twitter.com")

    def store_tweets_for_url(self, url: str, filename: str, limit: int = 1000, recusion_depth = 0) -> Tuple[List[Tweet], List[Tweet]]:
        parent_tweets, all_tweets = self.get_tweets_for_url(url, limit, recusion_depth)
        with open(filename, "w") as file:
            json.dump(parent_tweets, file, cls=TweetEncoder)
        return parent_tweets, all_tweets

    def load_tweets_from_file(self, filename: str) -> Tuple[List[Tweet], List[Tweet]]:
        with open(filename, "r") as file:
            json_list = json.load(file)
        tweets = [Tweet.from_dict(tweet) for tweet in json_list]
        return tweets, self.expand_parent_tweets(tweets)

    def get_tweets_for_url(self, url: str, limit: int = 1000, recusion_depth = 0) -> Tuple[List[Tweet], List[Tweet]]:
        print("Searching tweets with URL...", end=" ")
        parent_tweets = self.search(url, limit=limit)
        print("Done.\n")

        print("Finding related tweets...")

        related_tweets = []
        progress = tqdm(parent_tweets)
        progress.set_description("Overall Progress")

        for tweet in progress:
            related_tweets += self.get_related_tweets(tweet, limit, recusion_depth)
        related_tweets += parent_tweets

        print(f"Done.\n\nFound {len(related_tweets)} related tweets.")

        return parent_tweets, related_tweets

        # url_analyser = URLAnalyzer(url)
        # title = url_analyser.title
        # google_search = self.google.search(title, num=10)
        # for result in google_search:
        #     print(f"{result.title}: {result.url}")

    def new_twint_config(self, hide_output: bool = True) -> twint.Config:
        c = twint.Config()
        c.Hide_output = hide_output
        return c

    def search(self, search_term: str, limit: int = 10, since: str = None, hide_output: bool = True) -> List[Tweet]:
        c = self.new_twint_config(hide_output)
        c.Search = search_term
        c.Limit = limit
        if since: c.Since = since

        results = TwintCapturing.search(c)
        return Tweet.parse_twint(results, TweetOrigin.SEARCH)

    def get_replies(self, tweet: Tweet, limit: int = 1000) -> List[Tweet]:
        c = self.new_twint_config()
        c.To = tweet.mention
        c.Since = tweet.search_date
        c.Limit = limit

        results = TwintCapturing.search(c)
        if results.empty: return []

        filtered_results = results[results["conversation_id"] == tweet.conversation_id]
        return Tweet.parse_twint(filtered_results, TweetOrigin.REPLY)

    def get_retweets_twint(self, tweet: Tweet, limit: int = 1000) -> List[Tweet]:
        c = self.new_twint_config()
        c.All = tweet.mention
        c.Since = tweet.search_date
        c.Native_retweets = True
        c.Limit = limit

        results = TwintCapturing.search(c)
        if results.empty: return []

        filtered_results = results[results["retweet_id"] == tweet.tweet_id]
        return Tweet.parse_twint(filtered_results, TweetOrigin.REPLY)

    def get_retweets(self, tweet: Tweet, limit: int = 1000) -> List[Tweet]:
        try:
            results = self.api.get_retweets(tweet.tweet_id)
        except: # to bypass rate limit
            return self.get_retweets_twint(tweet, limit)
        return Tweet.parse_tweepy(results, TweetOrigin.RETWEET)

    def get_quotes_twint(self, tweet: Tweet, limit: int = 1000) -> List[Tweet]:
        return self.search(f"url:{tweet.tweet_id}", limit=limit)
        # return self.search(f"url:{tweet.tweet_id}", limit=limit, since=tweet.search_date)
    
    def get_quotes_tweepy(self, tweet: Tweet, limit: int = 10) -> List[Tweet]:
        # use twint instead, this is rate limited as hell
        response = self.client.get_quote_tweets(tweet.tweet_id, max_results=min(limit, 100))
        if not response.data: return []

        statuses = [self.api.get_status(tweet.id) for tweet in response.data]
        return Tweet.parse_tweepy(statuses, TweetOrigin.QUOTE)

    def get_related_tweets(self, tweet: Tweet, limit: int = 1000, recursion_depth: int = 0, ignore_ids: List[str] = []) -> List[Tweet]:

        def filter_tweets(tweets: List[Tweet]):
            return [tweet for tweet in tweets if tweet.tweet_id not in ignore_ids]

        replies  = filter_tweets(self.get_replies(tweet, limit))
        retweets = filter_tweets(self.get_retweets(tweet)) if tweet.stats.retweet_count > 0 else []
        quotes   = filter_tweets(self.get_quotes_twint(tweet, limit))

        tweet.add_replies(replies)
        tweet.add_retweets(retweets)
        tweet.add_quotes(quotes)

        tweets = replies + retweets + quotes
        ignore_ids += [tweet.tweet_id for tweet in tweets]

        results = []
        if recursion_depth > 0:

            progress = tqdm(tweets, leave=False)
            progress.set_description(f"Recursion Depth {recursion_depth}")

            for recursive_tweet in progress:
                results += self.get_related_tweets(recursive_tweet, limit, recursion_depth - 1, ignore_ids)

        return tweets + results

    def expand_parent_tweets(self, parent_tweets: List[Tweet]) -> List[Tweet]:
        if len(parent_tweets) == 0: return []

        children = []
        for tweet in parent_tweets:
            children += tweet.childern
        return parent_tweets + self.expand_parent_tweets(children)
    
    def tweets_to_dataframe(self, tweets: List[Tweet]) -> DataFrame:
        data = [tweet.dataframe_row for tweet in tweets]
        return DataFrame(data, columns=Tweet.DATAFRAME_COLUMS)

    def tweets_to_csv(self, tweets: List[Tweet], filename: str) -> None:
        df = self.tweets_to_dataframe(tweets)
        df.to_csv(filename)


Tweet = TwitterAnalyzer.Tweet
TweetOrigin = Tweet.Origin
TweetStats = Tweet.Stats
TweetEncoder = Tweet.TweetEncoder