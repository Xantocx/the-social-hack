from NewsTracker import Configuration
from NewsTracker.URLAnalyzer import URLAnalyzer

import tweepy
import twint
from pandas import DataFrame


class TwitterAnalyzer:

    def __init__(self, bearer_token: str) -> None:
        self.auth   = tweepy.OAuth2BearerHandler(bearer_token)
        self.client = tweepy.Client(bearer_token)
        self.api    = tweepy.API(self.auth)

    @classmethod
    def create_from(cls, config: Configuration):
        return TwitterAnalyzer(config.twitter_bearer_token)

    def analyze_url(self, url: str) -> None:
        url_analyser = URLAnalyzer(url)
        title = url_analyser.title
        print(title)

    def search(self, search_term: str, limit: int = 10, hide_output: bool = True) -> DataFrame:
        config = twint.Config()
        config.Search = search_term
        config.Limit = limit
        config.Pandas= True
        config.Hide_output = hide_output
        twint.run.Search(config)

        return twint.output.panda.Tweets_df