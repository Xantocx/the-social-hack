from NewsTracker import Configuration
from NewsTracker.URLAnalyzer import URLAnalyzer
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
from statistics import mean

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pandas import DataFrame, concat
from tqdm import tqdm


# HELPFUL TWINT REFERENCE:
# https://github.com/twintproject/twint/wiki/Configuration


class TwintCapturing(list):

    RESUME_LOG_FILE = "./temporary_resume_log.txt"

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout

    @classmethod
    def fast_search(cls, config: twint.Config, language: str = "en") -> DataFrame:

        if config.Search and language: config.Search += f" lang:{language}"
        config.Pandas = True

        with cls() as _:
            try:
                twint.run.Search(config)
            except:
                return DataFrame()
        return twint.output.panda.Tweets_df

    @classmethod
    def search(cls, config: twint.Config, language: str = "en", repetitions: int = 1) -> DataFrame:
        if repetitions < 2: return cls.fast_search(config)

        if config.Search and language: config.Search += f" lang:{language}"
        config.Pandas = True
        config.Resume = cls.RESUME_LOG_FILE

        results = [DataFrame()]
        for _ in range(repetitions):
            with cls() as _:
                try:
                    twint.run.Search(config)
                except:
                    break
            results.append(twint.output.panda.Tweets_df)

        if os.path.exists(cls.RESUME_LOG_FILE): os.remove(cls.RESUME_LOG_FILE)
        return concat(results)


class TwitterAnalyzer:


    class Tweet:

        SIA = SentimentIntensityAnalyzer() # vader sentiment analysis tool

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


        class TweetEncoder(JSONEncoder):
            def default(self, obj):
                return obj.__dict__ 


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

            self.tweet_id = tweet_id
            self.conversation_id = conversation_id
            self.username = username
            self.date = str(datetime.strptime(date, '%Y-%m-%d %H:%M:%S'))
            self.text = text
            self.url = url
            self._origin = origin.value

            # basic stats (still not sure if all retweets should be zero-ed)
            self.like_count    = like_count    if origin != TweetOrigin.RETWEET else 0
            self.reply_count   = reply_count   if origin != TweetOrigin.RETWEET else 0
            self.retweet_count = retweet_count if origin != TweetOrigin.RETWEET else 0
            self.quote_count   = quote_count   if origin != TweetOrigin.RETWEET else 0

            # sentiment analysis
            self.negative_sentiment = negative_sentiment
            self.neutral_sentiment  = neutral_sentiment
            self.positive_sentiment = positive_sentiment
            self.compound_sentiment = compound_sentiment

            self.replies  = []
            self.retweets = []
            self.quotes   = []

            # lacking data request
            # some data points cannot be provided by (all) APIs, so we approximate the value once we scraped for the according tweets
            self.request_reply_count = request_reply_count
            self.request_quote_count = request_quote_count

            self.update_sentiment()

        @property
        def mention(self) -> str:
            return f"@{self.username}"

        @property
        def hashtags(self) -> List[str]:
            return re.findall('(#[A-Za-z]+[A-Za-z0-9-_]+)', self.text)
        
        @property
        def clean_text(self) -> str:

            p.set_options(p.OPT.MENTION ,p.OPT.URL)

            clean_tweet = self.text.replace("#", "") # remove hashtag symbol
            clean_tweet = p.clean(clean_tweet)

            # Remove retweets:
            clean_tweet = re.sub(r'RT : ', '', clean_tweet)
            #remove amp
            clean_tweet = re.sub(r'&amp;', '', clean_tweet)
            #rempve strange characters
            clean_tweet = re.sub(r'ðŸ™', '', clean_tweet)
            #remove new lines
            clean_tweet = re.sub(r'\n', ' ', clean_tweet)

            return clean_tweet

        @property
        def origin(self) -> Origin:
            return TweetOrigin(self._origin)

        @property
        def engagement_score(self) -> int:
            return (self.like_count + 2 * self.retweet_count + 3 * self.reply_count + 4 * self.quote_count) / 10

        @property
        def is_positive(self) -> bool:
            if self.compound_sentiment:
                return self.compound_sentiment > 0
            return None

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

        def add_replies(self, tweets: List) -> None:
            self.replies += tweets
            if self.request_reply_count:
                self.reply_count = len(self.replies)

        def add_retweets(self, tweets: List) -> None:
            self.retweets += tweets

        def add_quotes(self, tweets: List) -> None:
            self.quotes += tweets
            if self.request_quote_count:
                self.quote_count = len(self.quotes)

        def update_sentiment(self) -> None:
            polarity = Tweet.SIA.polarity_scores(self.clean_text)
            if self.negative_sentiment is None: self.negative_sentiment = polarity["neg"]
            if self.neutral_sentiment  is None: self.neutral_sentiment  = polarity["neu"]
            if self.positive_sentiment is None: self.positive_sentiment = polarity["pos"]
            if self.compound_sentiment is None: self.compound_sentiment = polarity["compound"]

        def __repr__(self) -> str:
            return f"Tweet\n\tID: {self.tweet_id}\n\tUsername: {self.username}\n\tText: {self.text}\n\tURL: {self.url}\n"

        @classmethod
        def from_dict(cls, tweet_dict):

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
            else:
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

            tweet.add_replies([Tweet.from_dict(reply) for reply in tweet_dict["replies"]])
            tweet.add_retweets([Tweet.from_dict(retweet) for retweet in tweet_dict["retweets"]])
            tweet.add_quotes([Tweet.from_dict(quote) for quote in tweet_dict["quotes"]])

            return tweet

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


    class User:

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

            self.tweets   = []

        @property
        def tweet_count(self) -> int:
            return len(self.tweets)

        @property
        def likes(self) -> int:
            return sum(tweet.like_count for tweet in self.tweets)

        @property
        def replies(self) -> int:
            return sum(tweet.reply_count for tweet in self.tweets)

        @property
        def retweets(self) -> int:
            return sum(tweet.retweet_count for tweet in self.tweets)

        @property
        def quotes(self) -> int:
            return sum(tweet.quote_count for tweet in self.tweets)

        @property
        def engagement_score(self) -> int:
            return (self.likes + 2 * self.retweets + 3 * self.replies + 4 * self.quotes) / 10

        @property
        def negative_sentiment(self) -> float:
            return sum(tweet.negative_sentiment for tweet in self.tweets) / self.tweet_count

        @property
        def neutral_sentiment(self) -> float:
            return sum(tweet.neutral_sentiment for tweet in self.tweets) / self.tweet_count

        @property
        def positive_sentiment(self) -> float:
            return sum(tweet.neutral_sentiment for tweet in self.tweets) / self.tweet_count

        @property
        def compound_sentiment(self) -> float:
            return sum(tweet.compound_sentiment for tweet in self.tweets) / self.tweet_count

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

        def add_tweet(self, tweet) -> None:
            self.tweets.append(tweet)


    def __init__(self, config: Configuration) -> None:
        self.config = config
        token = config.twitter_bearer_token

        self.auth   = tweepy.OAuth2BearerHandler(token)
        self.client = tweepy.Client(token, wait_on_rate_limit=True)
        self.api    = tweepy.API(self.auth, wait_on_rate_limit=True)

        self.google = GoogleSearch(self.config, "twitter.com")

    def analyze_url(self, url: str, foldername: str) -> None:
        folder = foldername if foldername[-1] == "/" else foldername + "/"

        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "url.txt"), "w") as file:
            file.write(url)

        parent_tweets, all_tweets = self.store_tweets_for_url(url, os.path.join(folder, "tweets.json"))
        if parent_tweets is not None:
            self.analyze(parent_tweets, all_tweets, folder)

    def analyze_tweets_file(self, tweets_file: str, foldername: str) -> None:

        parent_tweets, all_tweets = self.load_tweets_from_file(tweets_file)

        # make sure tweets json file is in target folder as well
        folder = foldername if foldername[-1] == "/" else foldername + "/"
        source_file = os.path.abspath(tweets_file)
        target_file = os.path.abspath(os.path.join(folder, "tweets.json"))
        if source_file != target_file:
            os.makedirs(folder, exist_ok=True)
            shutil.copy(source_file, target_file)

        self.analyze(parent_tweets, all_tweets, folder)
        
    def analyze(self, parent_tweets: List[Tweet], all_tweets: List[Tweet], foldername: str) -> None:

        def filename(filename: str) -> str: return os.path.join(foldername, filename)

        print("\nStart analysis...\n")

        print("Extracting users...")
        users = self.get_users_from_tweets(all_tweets)

        print("Write metadata...")
        with open(filename("metadata.txt"), "w") as file:
            file.writelines([f"Tweet Count: {len(all_tweets)}\n",
                             f"User Count:  {len(users)}\n"])

        print("Generating CSVs...")
        self.tweets_to_csv(all_tweets, filename("tweets.csv"))
        self.users_to_csv(users, filename("users.csv"))
        
        print("Generating graphs...")
        # tweet graphs
        self.tweets_by_origin_graph(all_tweets, filename("tweets_by_origin.png"))
        self.tweets_by_sentiment_graph(all_tweets, filename("tweets_by_sentiment.png"))
        self.sentiment_by_origin_graph(all_tweets, filename("sentiment_by_origin.png"))
        self.engagement_by_origin_graph(all_tweets, filename("total_engagement_by_origin.png"), filename("avg_engagement_by_origin.png"))
        self.engagement_by_sentiment_tweets_graph(all_tweets, filename("total_engagement_by_sentiment_tweets.png"), filename("avg_engagement_by_sentiment_tweets.png"))

        # user graphs
        self.engagement_by_sentiment_users_graph(users, filename("total_engagement_by_sentiment_users.png"), filename("avg_engagement_by_sentiment_users.png"))
        self.user_scatter_log_graph(users, filename("followers_scatter_log.png"))
        self.user_scatter_filtered_graph(users, filename("followers_scatter_filtered.png"))

        print("\nDone with analysis.")

    def average_engagement(self, tweets: List[Tweet]) -> None:
        num_tweets = len(list(filter(lambda tweet: tweet.origin != TweetOrigin.RETWEET, tweets)))
        if num_tweets == 0: return 0 
        return sum(tweet.engagement_score for tweet in tweets) / num_tweets

    def tweets_by_origin(self, tweets: List[Tweet]) -> Tuple[List[str], List[List[Tweet]]]:
        origin_order = [origin.value for origin in TweetOrigin]
        return origin_order, [list(filter(lambda tweet: tweet._origin == origin, tweets)) for origin in origin_order]

    def tweets_by_sentiment(self, tweets: List[Tweet]) -> Tuple[List[float], List[List[Tweet]]]:
        sentiment_order = [x/10 for x in range(-10, 11)]
        return sentiment_order, [list(filter(lambda tweet: x-0.05 <= tweet.compound_sentiment < x+0.05, tweets)) for x in sentiment_order]

    def users_by_sentiment(self, users: Dict[str, User]) -> Tuple[List[float], List[List[User]]]:
        sentiment_order = [x/10 for x in range(-10, 11)]
        return sentiment_order, [list(filter(lambda user: x-0.05 <= user.compound_sentiment < x+0.05, users.values())) for x in sentiment_order]

    def tweets_by_origin_graph(self, tweets: List[Tweet], filename: str) -> None:
        origins_x = [origin.value for origin in TweetOrigin]
        origins_y = [len(list(filter(lambda tweet: tweet._origin == origin, tweets))) for origin in origins_x]

        plt.bar(origins_x, origins_y)
        plt.title("Tweets by Origin")
        plt.xlabel('Origin')
        plt.ylabel('Number of Tweets')
        plt.savefig(filename)
        plt.clf()

    def tweets_by_sentiment_graph(self, tweets: List[Tweet], filename: str) -> None:
        sentiment_x = [x/10 for x in range(-10, 11)]
        sentiment_y = [len(list(filter(lambda tweet: x-0.05 <= tweet.compound_sentiment < x+0.05, tweets))) for x in sentiment_x]

        plt.plot(sentiment_x, sentiment_y)
        plt.title("Tweets by Sentiment")
        plt.xlabel('Compound Sentiment')
        plt.ylabel('Number of Tweets')
        plt.savefig(filename)
        plt.clf()

    def sentiment_by_origin_graph(self, tweets: List[Tweet], filename: str) -> None:
        origins_x, tweets_by_origin = self.tweets_by_origin(tweets)
        sentiment_y = [sum(tweet.compound_sentiment for tweet in origin_tweets) / max(len(origin_tweets), 1) for origin_tweets in tweets_by_origin]

        plt.bar(origins_x, sentiment_y)
        plt.title("Sentiment by Origin")
        plt.xlabel('Origin')
        plt.ylabel('Compound Sentiment')
        plt.savefig(filename)
        plt.clf()

    def engagement_by_origin_graph(self, tweets: List[Tweet], total_filename: str, avg_filename: str) -> None:
        origin_x, tweets_by_origin = self.tweets_by_origin(tweets)

        total_engagement_y = [sum(tweet.engagement_score for tweet in origin_tweets) for origin_tweets in tweets_by_origin]
        avg_engagement_y   = [self.average_engagement(origin_tweets) for origin_tweets in tweets_by_origin]

        # based on sentiment, how high is engagement score
        plt.bar(origin_x, total_engagement_y)
        plt.title("Total Engagement by Origin")
        plt.xlabel('Origin')
        plt.ylabel('Total Engagement')
        plt.savefig(total_filename)
        plt.clf()

        # based on sentiment, how high is the average engagement score
        plt.bar(origin_x, avg_engagement_y)
        plt.title("Average Engagement by Origin")
        plt.xlabel('Origin')
        plt.ylabel('Average Engagement')
        plt.savefig(avg_filename)
        plt.clf()

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

    def store_tweets_for_url(self, url: str, filename: str, limit: int = 1000, recusion_depth = 0) -> Tuple[List[Tweet], List[Tweet]]:
        parent_tweets, all_tweets = self.get_tweets_for_url(url, limit, recusion_depth)
        self.store_tweets(parent_tweets, filename)
        return parent_tweets, all_tweets

    def get_tweets_for_url(self, url: str, limit: int = 1000, recusion_depth = 0) -> Tuple[List[Tweet], List[Tweet]]:
        return self.get_tweets(f"url:{url}", limit, recusion_depth)

    def get_tweets(self, search_term: str, limit: int = 1000, recusion_depth = 0) -> Tuple[List[Tweet], List[Tweet]]:
        print(f"Searching tweets using search term '{search_term}'...", end=" ")
        parent_tweets = self.search(search_term, limit=limit, repetitions=10)
        print("Done.\n")

        if len(parent_tweets) == 0:
            print("Could not find any tweets.\n")
            return None, None

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

    def get_users_from_tweets(self, tweets: List[Tweet]) -> Dict[str, User]:
        usernames = list(set([tweet.username for tweet in tweets]))
        requested_users = len(usernames)
        required_requests = ceil(requested_users / 100) if len(usernames) > 100 else 1

        user_info = []
        request_round = 0
        while request_round < required_requests:
            user_batch = usernames[request_round*100:min(request_round*100+100, requested_users)]
            user_info += self.api.lookup_users(screen_name=user_batch)
            request_round += 1

        users = {user.screen_name: TwitterUser(user.screen_name, user.followers_count) for user in user_info}

        deleted_users = 0
        for tweet in tweets:
            user = tweet.username
            if user in users:
                users[user].add_tweet(tweet)
            else:
                deleted_users += 1

        if deleted_users > 0: print(f"\n{deleted_users} user(s) deleted.\n")

        return users


    def new_twint_config(self, hide_output: bool = True) -> twint.Config:
        c = twint.Config()
        c.Hide_output = hide_output
        return c

    def search(self, search_term: str, 
                     limit: int = 1000, 
                     since: str = None, 
                     repetitions: int = 10,
                     origin: Tweet.Origin = Tweet.Origin.SEARCH, 
                     hide_output: bool = True) -> List[Tweet]:

        c = self.new_twint_config(hide_output)
        c.Search = search_term
        c.Limit = limit
        if since: c.Since = since

        results = TwintCapturing.search(c, repetitions=repetitions)
        return Tweet.parse_twint(results, origin)

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
        return self.search(f"url:{tweet.tweet_id}", limit=limit, origin=TweetOrigin.QUOTE, repetitions=1)
        # return self.search(f"url:{tweet.tweet_id}", limit=limit, since=tweet.search_date, origin=TweetOrigin.QUOTE, repetitions=1)
    
    def get_quotes_tweepy(self, tweet: Tweet, limit: int = 10) -> List[Tweet]:
        # use twint instead, this is rate limited as hell
        response = self.client.get_quote_tweets(tweet.tweet_id, max_results=min(limit, 100))
        if not response.data: return []

        statuses = [self.api.get_status(tweet.id) for tweet in response.data]
        return Tweet.parse_tweepy(statuses, TweetOrigin.QUOTE)

    def get_related_tweets(self, tweet: Tweet, limit: int = 1000, recursion_depth: int = 0, ignore_ids: List[str] = []) -> List[Tweet]:

        def filter_tweets(tweets: List[Tweet]):
            return [tweet for tweet in tweets if tweet.tweet_id not in ignore_ids]

        replies  = filter_tweets(self.get_replies(tweet, limit))      if tweet.origin != TweetOrigin.RETWEET else [] # performance optimation under the assumption that retweets don't have replies, as they are part of the original tweet
        retweets = filter_tweets(self.get_retweets(tweet))            if tweet.retweet_count > 0             else []
        quotes   = filter_tweets(self.get_quotes_twint(tweet, limit)) if tweet.origin != TweetOrigin.RETWEET else [] # performance optimation under the assumption that retweets don't have replies, as they are part of the original tweet

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
    
    def store_tweets(self, parent_tweets: List[Tweet], filename: str) -> None:
        if parent_tweets is None: return
        with open(filename, "w") as file:
            json.dump(parent_tweets, file, cls=TweetEncoder)

    def load_tweets_from_file(self, filename: str) -> Tuple[List[Tweet], List[Tweet]]:
        with open(filename, "r") as file:
            json_list = json.load(file)
        tweets = [Tweet.from_dict(tweet) for tweet in json_list]
        return tweets, self.expand_parent_tweets(tweets)

    def tweets_to_dataframe(self, tweets: List[Tweet]) -> DataFrame:
        data = [tweet.dataframe_row for tweet in tweets]
        return DataFrame(data, columns=Tweet.DATAFRAME_COLUMS)

    def tweets_to_csv(self, tweets: List[Tweet], filename: str) -> None:
        df = self.tweets_to_dataframe(tweets)
        df.to_csv(filename)

    def users_to_dataframe(self, users: Dict[str, User]) -> DataFrame:
        data = [user.dataframe_row for user in users.values()]
        return DataFrame(data, columns=TwitterUser.DATAFRAME_COLUMS)

    def users_to_csv(self, users: Dict[str, User], filename: str) -> None:
        df = self.users_to_dataframe(users)
        df.to_csv(filename)


Tweet = TwitterAnalyzer.Tweet
Tweets = List[Tweet]
TweetOrigin = Tweet.Origin
TweetEncoder = Tweet.TweetEncoder
TwitterUser = TwitterAnalyzer.User
TwitterUsers = List[TwitterUser]