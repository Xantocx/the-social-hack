class Configuration:

    GOOGLE_API_KEY   = "GOOGLE_API_KEY"
    SEARCH_ENGINE_ID = "CUSTOM_SEARCH_ENGINE_ID"
    TWITTER_BEARER_TOKEN = "TWITTER_BEARER_TOKEN"

    def __init__(self, google_api_key: str = None, 
                 search_engine_id:     str = None,
                 twitter_bearer_token: str = None):

        # Googl Keys         
        self.google_api_key:       str = google_api_key
        self.search_engine_id:     str = search_engine_id

        # Twitter Keys
        self.twitter_bearer_token: str = twitter_bearer_token

    @classmethod
    def load_from(cls, filename: str):
        with open(filename, "r") as file:
            key_value_pairs = [line.strip().split("=") for line in file.readlines() if len(line.strip()) >= 3]
            config = {key_value_pair[0]: key_value_pair[1] for key_value_pair in key_value_pairs if len(key_value_pair) == 2}

        return Configuration(google_api_key       = config[cls.GOOGLE_API_KEY], 
                             search_engine_id     = config[cls.SEARCH_ENGINE_ID],
                             twitter_bearer_token = config[cls.TWITTER_BEARER_TOKEN])
