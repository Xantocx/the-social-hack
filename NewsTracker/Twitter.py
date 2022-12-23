from NewsTracker import Configuration
from NewsTracker.Google import GoogleSearch

import tweepy
import twint
import json
import sys
import shutil
import os
import re
import matplotlib.pyplot as plt
import preprocessor as p

from typing import List, Tuple, Dict
from enum import Enum
from json import JSONEncoder
from datetime import datetime, timedelta
from io import StringIO
from math import ceil

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pandas import DataFrame, concat
from tqdm import tqdm


# HELPFUL TWINT REFERENCE:
# https://github.com/twintproject/twint/wiki/Configuration


# class to capture twint output in order to allow for proper progress bar
class TwintCapturing(list):

    # log file to store the current progress of twints scraping
    RESUME_LOG_FILE = "./temporary_resume_log.txt"

    # context manager entered: pipe stdout into void temporarily
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    # context manager left: restore stdout
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout

    # method to perform a single scrape on the twitter search page, using the provided config
    @classmethod
    def fast_search(cls, config: twint.Config, language: str = "en") -> DataFrame:

        # apply language filter, if needed
        if config.Search and language: config.Search += f" lang:{language}"
        config.Pandas = True

        # enter context manager to capture output and run config
        with cls() as _:
            try:
                twint.run.Search(config)
            except:
                return DataFrame()

        # return found tweets as pandas dataframe
        return twint.output.panda.Tweets_df

    # method to perform search on twitter with provided config, and searching for several repetitions to increase amount of tweets found
    @classmethod
    def search(cls, config: twint.Config, language: str = "en", repetitions: int = 1) -> DataFrame:
        # default to fast search if only one repetition is requested
        if repetitions < 2: return cls.fast_search(config)

        # apply language filter and set up continuous scraping
        if config.Search and language: config.Search += f" lang:{language}"
        config.Pandas = True
        config.Resume = cls.RESUME_LOG_FILE

        # scrape twitter based on config and collect search results
        results = [DataFrame()]
        for _ in range(repetitions):
            with cls() as _:
                try:
                    twint.run.Search(config)
                except:
                    break
            results.append(twint.output.panda.Tweets_df)

        # delete log file used to track scraping progress
        if os.path.exists(cls.RESUME_LOG_FILE): os.remove(cls.RESUME_LOG_FILE)
        # merge and return data frames
        return concat(results)


# class to perform full analysis of tweets for a URL
class TwitterAnalyzer:


    # tweet object holding all info that is read for a tweet
    class Tweet:

        # sentiment analyzer using vader for sentiment analysis
        SIA = SentimentIntensityAnalyzer() # vader sentiment analysis tool

        # column names for CSV of tweets
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
                            "Engagement Score",
                            "Negative Sentiment Score",
                            "Neutral Sentiment Score",
                            "Positive Sentiment Score",
                            "Compound Sentiment Score",
                            "Hashtags"]


        # JSON encoder for tweet serialization
        class TweetEncoder(JSONEncoder):
            def default(self, obj):
                return obj.__dict__ 


        # enum to simplify reference to tweet origin
        class Origin(Enum):
            SEARCH = "search"
            REPLY = "reply"
            RETWEET = "retweet"
            QUOTE = "quote"

        def __init__(self, tweet_id: str, 
                           conversation_id: str, 
                           username: str, 
                           date,
                           text: str, 
                           url: str,
                           origin: Origin,
                           like_count:          int,
                           reply_count:         int, 
                           retweet_count:       int,
                           quote_count:         int  = 0,
                           negative_sentiment:  bool = None,
                           neutral_sentiment:   bool = None,
                           positive_sentiment:  bool = None,
                           compound_sentiment:  bool = None,
                           request_reply_count: bool = False,
                           request_quote_count: bool = True) -> None:

            # set basic tweet properties
            self.tweet_id = tweet_id
            self.conversation_id = conversation_id
            self.username = username
            self.date = str(datetime.strptime(date, '%Y-%m-%d %H:%M:%S'))
            self.text = text
            self.url = url
            self._origin = origin.value

            # basic stats for tweets (still not sure if all retweets should be zero-ed)
            self.like_count    = like_count    if origin != TweetOrigin.RETWEET else 0
            self.reply_count   = reply_count   if origin != TweetOrigin.RETWEET else 0
            self.retweet_count = retweet_count if origin != TweetOrigin.RETWEET else 0
            self.quote_count   = quote_count   if origin != TweetOrigin.RETWEET else 0

            # sentiment analysis results
            self.negative_sentiment = negative_sentiment
            self.neutral_sentiment  = neutral_sentiment
            self.positive_sentiment = positive_sentiment
            self.compound_sentiment = compound_sentiment

            # related tweets
            self.replies  = []
            self.retweets = []
            self.quotes   = []

            # request counts for replies and quotes, if they cannot be read from API
            self.request_reply_count = request_reply_count
            self.request_quote_count = request_quote_count

            # update sentiment if necessary
            self.update_sentiment()

        # @username for the purpose of search in the analysis
        @property
        def mention(self) -> str:
            return f"@{self.username}"

        # hashtags extracted from tweet text
        @property
        def hashtags(self) -> List[str]:
            return re.findall('(#[A-Za-z]+[A-Za-z0-9-_]+)', self.text)
        
        # preprocessed text of tweet to simplify sentoment analysis
        @property
        def clean_text(self) -> str:

            # configure twitter preprocessor to remove mentions and urls
            p.set_options(p.OPT.MENTION ,p.OPT.URL)

            # remove hashtag symbol to keep hashtag word for sentiment analysis
            clean_tweet = self.text.replace("#", "") 
            # preprocess tweet
            clean_tweet = p.clean(clean_tweet)

            # Remove retweet symbols:
            clean_tweet = re.sub(r'RT : ', '', clean_tweet)
            #remove amp
            clean_tweet = re.sub(r'&amp;', '', clean_tweet)
            #rempve strange characters
            clean_tweet = re.sub(r'ðŸ™', '', clean_tweet)
            #remove new lines
            clean_tweet = re.sub(r'\n', ' ', clean_tweet)

            return clean_tweet

        # convenience accessor for origin object
        @property
        def origin(self) -> Origin:
            return TweetOrigin(self._origin)

        # calculate engagement score
        # this score is the weighted average of all possible interaction stats we can select for a tweet
        @property
        def engagement_score(self) -> int:
            return (self.like_count + 2 * self.retweet_count + 3 * self.reply_count + 4 * self.quote_count) / 10

        # when we search for tweets related to this one, we need to start the search one day before the tweet was posted to account for regional time shift in twitter's timestamps
        @property
        def search_date(self) -> str:
            return str(datetime.strptime(self.date, '%Y-%m-%d %H:%M:%S') - timedelta(days=1)).split()[0]

        # combination of all replies, retweets, and quotes
        @property
        def childern(self) -> List:
            return self.replies + self.retweets + self.quotes

        # converting this tweet into a single row in a dataframe for the purpose of writing it to a CSV
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
                self.like_count,
                self.reply_count,
                self.retweet_count,
                self.quote_count,
                self.engagement_score,
                self.negative_sentiment,
                self.neutral_sentiment,
                self.positive_sentiment,
                self.compound_sentiment,
                " ".join(self.hashtags)
            ]

        # add replies related to this tweet and update the reply count if necessary
        def add_replies(self, tweets: List) -> None:
            self.replies += tweets
            if self.request_reply_count:
                self.reply_count = len(self.replies)

        # add related retweets (we can always read the retweet count)
        def add_retweets(self, tweets: List) -> None:
            self.retweets += tweets

        # add quoted to this tweet and update the quote count if necessary
        def add_quotes(self, tweets: List) -> None:
            self.quotes += tweets
            if self.request_quote_count:
                self.quote_count = len(self.quotes)

        # update the sentiment analysis based on the preprocessed text for all values that have not been set before
        def update_sentiment(self) -> None:
            polarity = Tweet.SIA.polarity_scores(self.clean_text)
            if self.negative_sentiment is None: self.negative_sentiment = polarity["neg"]
            if self.neutral_sentiment  is None: self.neutral_sentiment  = polarity["neu"]
            if self.positive_sentiment is None: self.positive_sentiment = polarity["pos"]
            if self.compound_sentiment is None: self.compound_sentiment = polarity["compound"]

        # simplified string representation to print tweet in a meaningful way
        def __repr__(self) -> str:
            return f"Tweet\n\tID: {self.tweet_id}\n\tUsername: {self.username}\n\tText: {self.text}\n\tURL: {self.url}\n"

        # recover tweet from a dictionary that we recovered from a JSON file, basically a deserialization process
        @classmethod
        def from_dict(cls, tweet_dict):
            # legacy interface for JSON files generated before we restructured the interface
            if "stats" in tweet_dict:
                tweet = Tweet(
                    tweet_id            = tweet_dict["tweet_id"],
                    conversation_id     = tweet_dict["conversation_id"],
                    username            = tweet_dict["username"],
                    date                = tweet_dict["date"],
                    text                = tweet_dict["text"],
                    url                 = tweet_dict["url"],
                    origin              = TweetOrigin(tweet_dict["_origin"]),
                    like_count          = tweet_dict["stats"]["like_count"],
                    reply_count         = tweet_dict["stats"]["reply_count"],
                    retweet_count       = tweet_dict["stats"]["retweet_count"],
                    quote_count         = tweet_dict["stats"]["quote_count"],
                    negative_sentiment  = tweet_dict["stats"]["negative_sentiment"],
                    neutral_sentiment   = tweet_dict["stats"]["neutral_sentiment"],
                    positive_sentiment  = tweet_dict["stats"]["positive_sentiment"],
                    compound_sentiment  = tweet_dict["stats"]["compound_sentiment"],
                    request_reply_count = tweet_dict["stats"]["request_reply_count"],
                    request_quote_count = tweet_dict["stats"]["request_quote_count"]
                )
            else: # extraction of current JSON interface
                tweet = Tweet(
                    tweet_id            = tweet_dict["tweet_id"],
                    conversation_id     = tweet_dict["conversation_id"],
                    username            = tweet_dict["username"],
                    date                = tweet_dict["date"],
                    text                = tweet_dict["text"],
                    url                 = tweet_dict["url"],
                    origin              = TweetOrigin(tweet_dict["_origin"]),
                    like_count          = tweet_dict["like_count"],
                    reply_count         = tweet_dict["reply_count"],
                    retweet_count       = tweet_dict["retweet_count"],
                    quote_count         = tweet_dict["quote_count"],
                    negative_sentiment  = tweet_dict["negative_sentiment"],
                    neutral_sentiment   = tweet_dict["neutral_sentiment"],
                    positive_sentiment  = tweet_dict["positive_sentiment"],
                    compound_sentiment  = tweet_dict["compound_sentiment"],
                    request_reply_count = tweet_dict["request_reply_count"],
                    request_quote_count = tweet_dict["request_quote_count"]
                )

            # recursively read all replies, retweets and quotes from the JSON file and deserialize them as well
            tweet.add_replies([Tweet.from_dict(reply) for reply in tweet_dict["replies"]])
            tweet.add_retweets([Tweet.from_dict(retweet) for retweet in tweet_dict["retweets"]])
            tweet.add_quotes([Tweet.from_dict(quote) for quote in tweet_dict["quotes"]])

            return tweet

        # parse tweet info as provided by twint and convert it into a tweet object
        @classmethod
        def parse_twint(cls, tweet_data: DataFrame, origin: Origin):
            tweets = []
            for _, tweet_info in tweet_data.iterrows():
                tweet = Tweet(
                    tweet_id        = tweet_info["id"],
                    conversation_id = tweet_info["conversation_id"],
                    username        = tweet_info["username"],
                    date            = tweet_info["date"],
                    text            = tweet_info["tweet"],
                    url             = tweet_info["link"],
                    origin          = origin,
                    like_count      = tweet_info["nlikes"],
                    reply_count     = tweet_info["nreplies"],
                    retweet_count   = tweet_info["nretweets"]
                )
                tweets.append(tweet)
            return tweets

        # parse tweet info as provided by tweepy and convert it into a tweet object
        @classmethod
        def parse_tweepy(cls, statuses, origin: Origin):
            tweets = []
            for status in statuses:
                tweet = Tweet(
                    tweet_id            = status.id,
                    conversation_id     = status.in_reply_to_status_id,
                    username            = status.user.screen_name,
                    date                = str(status.created_at).split("+")[0],
                    text                = status.text,
                    url                 = f"https://twitter.com/twitter/statuses/{status.id}",
                    origin              = origin,
                    like_count          = status.favorite_count,
                    reply_count         = 0,
                    retweet_count       = status.retweet_count,
                    request_reply_count = True
                )
                tweets.append(tweet)
            return tweets


    # user class containing all the data we gather about the user, makin it available for analysis
    class User:

        # names for columns for the user CSV
        DATAFRAME_COLUMS = ["Username", 
                            "Followers", 
                            "Tweet Count", 
                            "Likes", 
                            "Replies", 
                            "Retweets",
                            "Quotes",
                            "Engagement Score",
                            "Negative Sentiment Score",
                            "Neutral Sentiment Score",
                            "Positive Sentiment Score",
                            "Compound Sentiment Score"]

        def __init__(self, username: str, followers: int) -> None:
            self.username  = username
            self.followers = followers

            # tweets the user made
            self.tweets   = []

        # number of tweets the user made
        @property
        def tweet_count(self) -> int:
            return len(self.tweets)

        # combined number of likes the user's tweets generated
        @property
        def likes(self) -> int:
            return sum(tweet.like_count for tweet in self.tweets)

        # combined number of replies the user's tweets generated
        @property
        def replies(self) -> int:
            return sum(tweet.reply_count for tweet in self.tweets)

        # combined number of retweets the user's tweets generated
        @property
        def retweets(self) -> int:
            return sum(tweet.retweet_count for tweet in self.tweets)

        # combined number of quotes the user's tweets generated
        @property
        def quotes(self) -> int:
            return sum(tweet.quote_count for tweet in self.tweets)

        # weighted engagement score for the user, based on the combined likes, retweets, replies, and quotes
        @property
        def engagement_score(self) -> int:
            return (self.likes + 2 * self.retweets + 3 * self.replies + 4 * self.quotes) / 10

        # average negative sentiment of the user's tweets
        @property
        def negative_sentiment(self) -> float:
            return sum(tweet.negative_sentiment for tweet in self.tweets) / self.tweet_count

        # average neutral sentiment of the user's tweets
        @property
        def neutral_sentiment(self) -> float:
            return sum(tweet.neutral_sentiment for tweet in self.tweets) / self.tweet_count

        # average positive sentiment of the user's tweets
        @property
        def positive_sentiment(self) -> float:
            return sum(tweet.neutral_sentiment for tweet in self.tweets) / self.tweet_count

        # average compound sentiment of the user's tweets
        @property
        def compound_sentiment(self) -> float:
            return sum(tweet.compound_sentiment for tweet in self.tweets) / self.tweet_count

        # generate a row for a dataframe from this user object to convert it into a CSV
        @property
        def dataframe_row(self) -> List:
            return [
                self.username,
                self.followers,
                self.tweet_count,
                self.likes,
                self.replies,
                self.retweets,
                self.quotes,
                self.engagement_score,
                self.negative_sentiment,
                self.neutral_sentiment,
                self.positive_sentiment,
                self.compound_sentiment
            ]

        # add tweet made by the user
        def add_tweet(self, tweet) -> None:
            self.tweets.append(tweet)


    def __init__(self, config: Configuration) -> None:
        # read tokens for API from config
        self.config = config
        token = config.twitter_bearer_token

        # generate objects to use tweepy API
        self.auth   = tweepy.OAuth2BearerHandler(token)
        self.client = tweepy.Client(token, wait_on_rate_limit=True)
        self.api    = tweepy.API(self.auth, wait_on_rate_limit=True)

    # take URL, find related tweets, and perform full analysis
    def analyze_url(self, url: str, foldername: str) -> None:
        # create valid foldername for this analysis
        folder = foldername if foldername[-1] == "/" else foldername + "/"

        # crate folder and write text file containing the analyzed URL
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "url.txt"), "w") as file:
            file.write(url)

        # find all related tweets and if there are any, analyze them
        parent_tweets, all_tweets = self.store_tweets_for_url(url, os.path.join(folder, "tweets.json"))
        if parent_tweets is not None:
            self.analyze(parent_tweets, all_tweets, folder)

    # take a JSON file containing serialized tweets and analyze them
    def analyze_tweets_file(self, tweets_file: str, foldername: str) -> None:

        # load tweets form JSON file
        parent_tweets, all_tweets = self.load_tweets_from_file(tweets_file)

        # copy tweets JSON file to target folder for the analysis
        folder = foldername if foldername[-1] == "/" else foldername + "/"
        source_file = os.path.abspath(tweets_file)
        target_file = os.path.abspath(os.path.join(folder, "tweets.json"))
        if source_file != target_file:
            os.makedirs(folder, exist_ok=True)
            shutil.copy(source_file, target_file)

        # perform analysis
        self.analyze(parent_tweets, all_tweets, folder)
        
    # analyze a set of tweets and generate several graphs for easy interpretation of data
    def analyze(self, parent_tweets: List[Tweet], all_tweets: List[Tweet], foldername: str) -> None:

        # custom function to generate a filename within the analysis folder
        def filename(filename: str) -> str: return os.path.join(foldername, filename)

        print("\nStart analysis...\n")

        # extract all users from the tweets
        print("Extracting users...")
        users = self.get_users_from_tweets(all_tweets)

        # write metadata text file containing count of tweets, as well as users
        print("Write metadata...")
        with open(filename("metadata.txt"), "w") as file:
            file.writelines([f"Tweet Count: {len(all_tweets)}\n",
                             f"User Count:  {len(users)}\n"])

        # generate CSV for tweets and users
        print("Generating CSVs...")
        self.tweets_to_csv(all_tweets, filename("tweets.csv"))
        self.users_to_csv(users, filename("users.csv"))
        
        print("Generating graphs...")

        # generate graphs based on tweet data
        self.tweets_by_origin_graph(all_tweets, filename("tweets_by_origin.png"))
        self.tweets_by_sentiment_graph(all_tweets, filename("tweets_by_sentiment.png"))
        self.sentiment_by_origin_graph(all_tweets, filename("sentiment_by_origin.png"))
        self.engagement_by_origin_graph(all_tweets, filename("total_engagement_by_origin.png"), filename("avg_engagement_by_origin.png"))
        self.engagement_by_sentiment_tweets_graph(all_tweets, filename("total_engagement_by_sentiment_tweets.png"), filename("avg_engagement_by_sentiment_tweets.png"))

        # generate graphs based on user data
        self.engagement_by_sentiment_users_graph(users, filename("total_engagement_by_sentiment_users.png"), filename("avg_engagement_by_sentiment_users.png"))
        self.user_scatter_log_graph(users, filename("followers_scatter_log.png"))
        self.user_scatter_filtered_graph(users, filename("followers_scatter_filtered.png"))

        print("\nDone with analysis.")

    # calculate the average engagement for a set of tweets
    # this requires a specific function, as we want to ignore retweets from this average
    # this is necessary, as retweets do not contribute to the engagement score, as they only mirror engagement on the original tweet
    def average_engagement(self, tweets: List[Tweet]) -> None:
        num_tweets = len(list(filter(lambda tweet: tweet.origin != TweetOrigin.RETWEET, tweets)))
        if num_tweets == 0: return 0 
        return sum(tweet.engagement_score for tweet in tweets) / num_tweets

    # sorts the tweets by origin, and returns both, a list of the origin and the tweets, in matching order, for the use with matplotlib
    def tweets_by_origin(self, tweets: List[Tweet]) -> Tuple[List[str], List[List[Tweet]]]:
        origin_order = [origin.value for origin in TweetOrigin]
        return origin_order, [list(filter(lambda tweet: tweet._origin == origin, tweets)) for origin in origin_order]

    # groups and sorts the tweets by sentiment, and returns both, a list of the sentiment groups and the tweets, in matching order, for the use with matplotlib
    def tweets_by_sentiment(self, tweets: List[Tweet]) -> Tuple[List[float], List[List[Tweet]]]:
        sentiment_order = [x/10 for x in range(-10, 11)]
        return sentiment_order, [list(filter(lambda tweet: x-0.05 <= tweet.compound_sentiment < x+0.05, tweets)) for x in sentiment_order]

    # groups and sorts the tweets by sentiment, and returns both, a list of the sentiment groups and the tweets, in matching order, for the use with matplotlib
    def users_by_sentiment(self, users: Dict[str, User]) -> Tuple[List[float], List[List[User]]]:
        sentiment_order = [x/10 for x in range(-10, 11)]
        return sentiment_order, [list(filter(lambda user: x-0.05 <= user.compound_sentiment < x+0.05, users.values())) for x in sentiment_order]

    # generate a bar chart displaying the number of tweets found by origin
    def tweets_by_origin_graph(self, tweets: List[Tweet], filename: str) -> None:
        origins_x = [origin.value for origin in TweetOrigin]
        origins_y = [len(list(filter(lambda tweet: tweet._origin == origin, tweets))) for origin in origins_x]

        plt.bar(origins_x, origins_y)
        plt.title("Tweets by Origin")
        plt.xlabel('Origin')
        plt.ylabel('Number of Tweets')
        plt.savefig(filename)
        plt.clf()

    # generate a plot showing how many tweets for a given sentiment were found
    # the sentiment is grouped in groups of size 0.1
    def tweets_by_sentiment_graph(self, tweets: List[Tweet], filename: str) -> None:
        sentiment_x = [x/10 for x in range(-10, 11)]
        sentiment_y = [len(list(filter(lambda tweet: x-0.05 <= tweet.compound_sentiment < x+0.05, tweets))) for x in sentiment_x]

        plt.plot(sentiment_x, sentiment_y)
        plt.title("Tweets by Sentiment")
        plt.xlabel('Compound Sentiment')
        plt.ylabel('Number of Tweets')
        plt.savefig(filename)
        plt.clf()

    # generates a bar chart showing the averave sentiment for tweets, grouped by origin
    def sentiment_by_origin_graph(self, tweets: List[Tweet], filename: str) -> None:
        origins_x, tweets_by_origin = self.tweets_by_origin(tweets)
        sentiment_y = [sum(tweet.compound_sentiment for tweet in origin_tweets) / max(len(origin_tweets), 1) for origin_tweets in tweets_by_origin]

        plt.bar(origins_x, sentiment_y)
        plt.title("Sentiment by Origin")
        plt.xlabel('Origin')
        plt.ylabel('Compound Sentiment')
        plt.savefig(filename)
        plt.clf()

    # generates two bar charts displaying the engagement for tweets grouped by origin, once using the summed engagement, and once the average engagement
    def engagement_by_origin_graph(self, tweets: List[Tweet], total_filename: str, avg_filename: str) -> None:
        origin_x, tweets_by_origin = self.tweets_by_origin(tweets)

        total_engagement_y = [sum(tweet.engagement_score for tweet in origin_tweets) for origin_tweets in tweets_by_origin]
        avg_engagement_y   = [self.average_engagement(origin_tweets) for origin_tweets in tweets_by_origin]

        # based on origin, how high is engagement score
        plt.bar(origin_x, total_engagement_y)
        plt.title("Total Engagement by Origin")
        plt.xlabel('Origin')
        plt.ylabel('Total Engagement')
        plt.savefig(total_filename)
        plt.clf()

        # based on origin, how high is the average engagement score
        plt.bar(origin_x, avg_engagement_y)
        plt.title("Average Engagement by Origin")
        plt.xlabel('Origin')
        plt.ylabel('Average Engagement')
        plt.savefig(avg_filename)
        plt.clf()

    # generates two bar charts displaying the engagement for tweets grouped by sentiment, once using the summed engagement, and once the average engagement
    def engagement_by_sentiment_tweets_graph(self, tweets: List[Tweet], total_filename: str, avg_filename: str) -> None:
        sentiment_x, tweets_by_sentiment = self.tweets_by_sentiment(tweets)
        total_engagement_y = [sum(tweet.engagement_score for tweet in sentiment_tweets) for sentiment_tweets in tweets_by_sentiment]
        avg_engagement_y   = [self.average_engagement(sentiment_tweets) for sentiment_tweets in tweets_by_sentiment]

        # based on sentiment, how high is engagement score
        plt.plot(sentiment_x, total_engagement_y)
        plt.title("Total Tweet Engagement by Sentiment")
        plt.xlabel('Compound Sentiment')
        plt.ylabel('Combined Engagement')
        plt.savefig(total_filename)
        plt.clf()

        # based on sentiment, how high is the average engagement score
        plt.plot(sentiment_x, avg_engagement_y)
        plt.title("Average Tweet Engagement by Sentiment")
        plt.xlabel('Compound Sentiment')
        plt.ylabel('Average Engagement')
        plt.savefig(avg_filename)
        plt.clf()

    # generates two bar charts displaying the engagement for users grouped by origin, once using the summed engagement, and once the average engagement
    def engagement_by_sentiment_users_graph(self, users: Dict[str, User], total_filename: str, avg_filename: str) -> None:
        sentiment_x, users_by_sentiment = self.users_by_sentiment(users)
        total_engagement_y = [sum(user.engagement_score for user in sentiment_users) for sentiment_users in users_by_sentiment]
        avg_engagement_y   = [sum(user.engagement_score for user in sentiment_users) / max(len(sentiment_users), 1) for sentiment_users in users_by_sentiment]

        # based on sentiment, how high is engagement score
        plt.plot(sentiment_x, total_engagement_y)
        plt.title("Total User Engagement by Sentiment")
        plt.xlabel('Compound Sentiment')
        plt.ylabel('Combined Engagement')
        plt.savefig(total_filename)
        plt.clf()

        # based on sentiment, how high is the average engagement score
        plt.plot(sentiment_x, avg_engagement_y)
        plt.title("Average User Engagement by Sentiment")
        plt.xlabel('Compound Sentiment')
        plt.ylabel('Average Engagement')
        plt.savefig(avg_filename)
        plt.clf()

    # generates a scatter plot, containing one point per user, with axis in logarithmic scale
    # the size of the dot corresponds to the number of tweets this user performed
    # the color represents the sentimend (red = negative, black = neutral, green = positive; it's a gradient)
    # the x axis corresponds to the amount of followers the user has
    # the x axis represents the engagement score of the user
    def user_scatter_log_graph(self, users: Dict[str, User], filename: str) -> None:
        user_list = users.values()
        engagmenent_y = [user.engagement_score for user in user_list]

        tweet_counts = [user.tweet_count for user in user_list]
        followers = [user.followers for user in user_list]

        max_tweets = max(tweet_counts)
        area_tweets = [max(1000 * tweet_count / max_tweets, 5) for tweet_count in tweet_counts]

        sentiment_colors = [(max(1.0 - (user.compound_sentiment + 1), 0), max(user.compound_sentiment, 0), 0.0) for user in user_list]
        
        # followers by engagement (log)
        plt.scatter(followers, engagmenent_y, s=area_tweets, c=sentiment_colors, alpha=0.5)
        plt.xscale("log")
        plt.yscale("log")
        plt.title("Scatter Plot (size = Tweet Count, color = Sentiment)")
        plt.xlabel('Follower Count (log)')
        plt.ylabel('Enagement Score (log)')
        plt.savefig(filename)
        plt.clf()

    # generates a scatter plot, containing one point per user, however we cut outliers to scale the plot in a reasonable area
    # the size of the dot corresponds to the number of tweets this user performed
    # the color represents the sentimend (red = negative, black = neutral, green = positive; it's a gradient)
    # the x axis corresponds to the amount of followers the user has
    # the x axis represents the engagement score of the user
    def user_scatter_filtered_graph(self, users: Dict[str, User], filename: str) -> None:
        user_list = users.values()
        engagement_limit = sorted(user.engagement_score for user in user_list)[int(0.98 * len(user_list))]
        tweets_limit = sorted(user.tweet_count for user in user_list)[int(1 * len(user_list) - 1)]
        followers_limit = sorted(user.followers for user in user_list)[int(0.95 * len(user_list))]

        user_list = list(filter(lambda user: user.engagement_score <= engagement_limit and
                                             user.tweet_count      <= tweets_limit and
                                             user.followers        <= followers_limit, users.values()))
        engagmenent_y = [user.engagement_score for user in user_list]

        tweet_counts = [user.tweet_count for user in user_list]
        followers = [user.followers for user in user_list]

        max_tweets = max(tweet_counts)
        area_tweets = [max(200 * tweet_count / max_tweets, 5) for tweet_count in tweet_counts]

        sentiment_colors = [(max(1.0 - (user.compound_sentiment + 1), 0), max(user.compound_sentiment, 0), 0.0) for user in user_list]

        # followers by engagement
        plt.scatter(followers, engagmenent_y, s=area_tweets, c=sentiment_colors, alpha=0.5)
        plt.title("Scatter Plot (size = Tweet Count, color = Sentiment)")
        plt.xlabel('Follower Count')
        plt.ylabel('Enagement Score')
        plt.savefig(filename)
        plt.clf()

    # finds tweets related to a URL and serializes them
    def store_tweets_for_url(self, url: str, filename: str, limit: int = 1000, recusion_depth = 0) -> Tuple[List[Tweet], List[Tweet]]:
        parent_tweets, all_tweets = self.get_tweets_for_url(url, limit, recusion_depth)
        self.store_tweets(parent_tweets, filename)
        return parent_tweets, all_tweets

    # generatate related tweets for a URL, applying the "url:" search modifier
    def get_tweets_for_url(self, url: str, limit: int = 1000, recusion_depth = 0) -> Tuple[List[Tweet], List[Tweet]]:
        return self.get_tweets(f"url:{url}", limit, recusion_depth)

    # find all related tweets to a given search term, including retweets, replies, and quotes
    # this search can be performed in a recursive manner if desired
    def get_tweets(self, search_term: str, limit: int = 1000, recusion_depth = 0) -> Tuple[List[Tweet], List[Tweet]]:
        # search tweets including the search term
        # these tweets provide the inital base of our data set
        print(f"Searching tweets using search term '{search_term}'...", end=" ")
        parent_tweets = self.search(search_term, limit=limit, repetitions=10)
        print("Done.\n")

        # if there is no tweets to work with, finish
        if len(parent_tweets) == 0:
            print("Could not find any tweets.\n")
            return None, None

        print("Finding related tweets...")

        # create object to display progress bar
        related_tweets = []
        progress = tqdm(parent_tweets)
        progress.set_description("Overall Progress")

        # for each tweet we found so far, find all related tweets recusively and add them to our tweet list
        for tweet in progress:
            related_tweets += self.get_related_tweets(tweet, limit, recusion_depth)
        related_tweets += parent_tweets  # add parent tweets to list as well

        print(f"Done.\n\nFound {len(related_tweets)} related tweets.")

        # return all tweets
        return parent_tweets, related_tweets

    # for a list of tweets, extract all users that made a tweet
    def get_users_from_tweets(self, tweets: List[Tweet]) -> Dict[str, User]:
        # extract all unique usernames from tweets
        usernames = list(set([tweet.username for tweet in tweets]))

        # generate counters needed to find additional info
        requested_users = len(usernames)
        required_requests = ceil(requested_users / 100) if len(usernames) > 100 else 1

        user_info = []
        request_round = 0
        
        # as each call of the twitter api can only return info for 100 users:
        # divide usernames in batches of size 100 and load additional info with twitter api
        while request_round < required_requests:
            # generate current batch
            user_batch = usernames[request_round*100:min(request_round*100+100, requested_users)]
            # reat user info
            user_info += self.api.lookup_users(screen_name=user_batch)
            request_round += 1

        # generate user object from user infos
        users = {user.screen_name: TwitterUser(user.screen_name, user.followers_count) for user in user_info}

        # add tweets to their respective author's user object
        deleted_users = 0
        for tweet in tweets:
            user = tweet.username
            if user in users:
                users[user].add_tweet(tweet)
            else:
                deleted_users += 1

        # print the number of users that were deleted and are missing accordingly
        if deleted_users > 0: print(f"\n{deleted_users} user(s) deleted.\n")

        return users

    # generate a new config object for twint with a config baseline configuration
    def new_twint_config(self, hide_output: bool = True) -> twint.Config:
        c = twint.Config()
        c.Hide_output = hide_output
        return c

    # search tweets with a search term on twitter and return them as tweet objects
    def search(self, search_term: str, 
                     limit: int = 1000, 
                     since: str = None, 
                     repetitions: int = 10,
                     origin: Tweet.Origin = Tweet.Origin.SEARCH, 
                     hide_output: bool = True) -> List[Tweet]:

        # setup config for twint
        c = self.new_twint_config(hide_output)
        c.Search = search_term
        c.Limit = limit
        if since: c.Since = since

        # search tweets and convert them
        results = TwintCapturing.search(c, repetitions=repetitions)
        return Tweet.parse_twint(results, origin)

    # use twint to find all replies to a specific tweet
    def get_replies(self, tweet: Tweet, limit: int = 1000) -> List[Tweet]:
        # setup twint to find tweets directed to the author or our respective tweet
        # make sure the found tweets were created after the baseline tweet
        c = self.new_twint_config()
        c.To = tweet.mention
        c.Since = tweet.search_date
        c.Limit = limit

        # search tweets
        results = TwintCapturing.search(c)
        if results.empty: return []

        # filter tweets that are sent in reply to the baseline tweet
        filtered_results = results[results["conversation_id"] == tweet.conversation_id]
        return Tweet.parse_twint(filtered_results, TweetOrigin.REPLY)

    # use twint to find all retweets of a baseline tweet
    def get_retweets_twint(self, tweet: Tweet, limit: int = 1000) -> List[Tweet]:
        # setup twint to find all tweets that include a mention to the author of out baeline tweet (retweets always do so)
        # filter for retweets only set were created after the baseline tweet
        c = self.new_twint_config()
        c.All = tweet.mention
        c.Since = tweet.search_date
        c.Native_retweets = True
        c.Limit = limit

        # search tweets
        results = TwintCapturing.search(c)
        if results.empty: return []

        # filter tweets that actually retweet the baseline tweet
        filtered_results = results[results["retweet_id"] == tweet.tweet_id]
        return Tweet.parse_twint(filtered_results, TweetOrigin.REPLY)

    # use tweepy to find retweets (faster than twint), and default to using twint when the rate limit is reached
    def get_retweets(self, tweet: Tweet, limit: int = 1000) -> List[Tweet]:
        try:
            # try to find retweets with tweepy
            results = self.api.get_retweets(tweet.tweet_id)
        except:
            # if that fails, use twint instead
            return self.get_retweets_twint(tweet, limit)
        return Tweet.parse_tweepy(results, TweetOrigin.RETWEET)

    # use the normal twint search to find all tweets quoting the URL to our baseline tweet -> these are quotes
    def get_quotes_twint(self, tweet: Tweet, limit: int = 1000) -> List[Tweet]:
        return self.search(f"url:{tweet.tweet_id}", limit=limit, origin=TweetOrigin.QUOTE, repetitions=1)
        # return self.search(f"url:{tweet.tweet_id}", limit=limit, since=tweet.search_date, origin=TweetOrigin.QUOTE, repetitions=1)
    
    # use tweepy to find quote tweets (not used due to rate limit)
    def get_quotes_tweepy(self, tweet: Tweet, limit: int = 10) -> List[Tweet]:
        response = self.client.get_quote_tweets(tweet.tweet_id, max_results=min(limit, 100))
        if not response.data: return []

        statuses = [self.api.get_status(tweet.id) for tweet in response.data]
        return Tweet.parse_tweepy(statuses, TweetOrigin.QUOTE)

    # use the above methods to find replies, retweets, or quotes to find all related tweets for a baseline tweet
    def get_related_tweets(self, tweet: Tweet, limit: int = 1000, recursion_depth: int = 0, ignore_ids: List[str] = []) -> List[Tweet]:

        # method filtering tweets that we already found
        def filter_tweets(tweets: List[Tweet]):
            return [tweet for tweet in tweets if tweet.tweet_id not in ignore_ids]

        # find related replies, retweets and quotes
        replies  = filter_tweets(self.get_replies(tweet, limit))      if tweet.origin != TweetOrigin.RETWEET else [] # performance optimation under the assumption that retweets don't have replies, as they are part of the original tweet
        retweets = filter_tweets(self.get_retweets(tweet))            if tweet.retweet_count > 0             else []
        quotes   = filter_tweets(self.get_quotes_twint(tweet, limit)) if tweet.origin != TweetOrigin.RETWEET else [] # performance optimation under the assumption that retweets don't have replies, as they are part of the original tweet

        # add related tweets to baseline tweet
        tweet.add_replies(replies)
        tweet.add_retweets(retweets)
        tweet.add_quotes(quotes)

        # prepare for recursive search
        tweets = replies + retweets + quotes
        ignore_ids += [tweet.tweet_id for tweet in tweets]

        # recursively find related tweets for all related tweets (this can for example find replies to replies, and quotes of quotes)
        # this is really slow, but effective -> in reality, we found enough data without this
        results = []
        if recursion_depth > 0:

            # setup secondary progress bar
            progress = tqdm(tweets, leave=False)
            progress.set_description(f"Recursion Depth {recursion_depth}")

            for recursive_tweet in progress:
                results += self.get_related_tweets(recursive_tweet, limit, recursion_depth - 1, ignore_ids)

        # return combined findings
        return tweets + results

    # from a set of parent tweets, create a list of all tweets in the data set
    # needed after deserializing a JSON file, which only contains the parents, which contain their children
    def expand_parent_tweets(self, parent_tweets: List[Tweet]) -> List[Tweet]:
        if len(parent_tweets) == 0: return []

        # recursively construct a list of all tweets
        children = []
        for tweet in parent_tweets:
            children += tweet.childern
        return parent_tweets + self.expand_parent_tweets(children)
    
    # serialize a list of parent tweets to JSON
    def store_tweets(self, parent_tweets: List[Tweet], filename: str) -> None:
        if parent_tweets is None: return
        with open(filename, "w") as file:
            json.dump(parent_tweets, file, cls=TweetEncoder)

    # load a list of parent tweets, plus the expanded list of tweets from a JSON file
    def load_tweets_from_file(self, filename: str) -> Tuple[List[Tweet], List[Tweet]]:
        with open(filename, "r") as file:
            json_list = json.load(file)
        tweets = [Tweet.from_dict(tweet) for tweet in json_list]
        return tweets, self.expand_parent_tweets(tweets)

    # create a dataframe containing all provided tweets
    def tweets_to_dataframe(self, tweets: List[Tweet]) -> DataFrame:
        data = [tweet.dataframe_row for tweet in tweets]
        return DataFrame(data, columns=Tweet.DATAFRAME_COLUMS)

    # write a CSV including data for all provided tweets
    def tweets_to_csv(self, tweets: List[Tweet], filename: str) -> None:
        df = self.tweets_to_dataframe(tweets)
        df.to_csv(filename)

    # create a dataframe containing all provided users
    def users_to_dataframe(self, users: Dict[str, User]) -> DataFrame:
        data = [user.dataframe_row for user in users.values()]
        return DataFrame(data, columns=TwitterUser.DATAFRAME_COLUMS)

    # write a CSV including data for all provided users
    def users_to_csv(self, users: Dict[str, User], filename: str) -> None:
        df = self.users_to_dataframe(users)
        df.to_csv(filename)

# typealiases for simplified type access
Tweet = TwitterAnalyzer.Tweet
Tweets = List[Tweet]
TweetOrigin = Tweet.Origin
TweetEncoder = Tweet.TweetEncoder
TwitterUser = TwitterAnalyzer.User
TwitterUsers = List[TwitterUser]