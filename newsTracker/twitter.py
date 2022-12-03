from newsTracker import Configuration

import tweepy
import twint

class TwitterAnalyzer:

    def __init__(self, bearer_token: str) -> None:
        self.auth   = tweepy.OAuth2BearerHandler(bearer_token)
        self.client = tweepy.Client(bearer_token)
        self.api    = tweepy.API(self.auth)

    @classmethod
    def create_from(cls, config: Configuration):
        return TwitterAnalyzer(config.twitter_bearer_token)

    def search(self, search_term: str, limit: int = 10, tmp_file: str = "tmp.json"):
        config = twint.Config()

        config.Search = search_term
        config.Limit = limit
        config.Store_json = True
        config.Output = tmp_file

        twint.run.Search(config)