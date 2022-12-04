from NewsTracker import Configuration

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
        # NOTE: THIS CLASS IS WORK IN PROGRESS
        # We need some way to elegantly read the data from the temporary file and work with it poperly. Maybe we can also delete the temporary file? Not sure if that is desirable.
        config = twint.Config()

        config.Search = search_term
        config.Limit = limit
        config.Store_json = True
        config.Output = tmp_file

        twint.run.Search(config)