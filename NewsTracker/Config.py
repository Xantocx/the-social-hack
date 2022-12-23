from collections import defaultdict


# class used to manage API keys for all other analyzers used in this project
# it reads the data from a file that will not be commited to the git repo, for safety reasons
class Configuration:

    # Google Keys: https://stackoverflow.com/questions/37083058/programmatically-searching-google-in-python-using-custom-search
    GOOGLE_API_KEY   = "GOOGLE_API_KEY"
    SEARCH_ENGINE_ID = "CUSTOM_SEARCH_ENGINE_ID"

    # Twitter Keys
    TWITTER_BEARER_TOKEN = "TWITTER_BEARER_TOKEN"

    # Reddit Keys
    REDDIT_USERNAME = "REDDIT_USERNAME"
    REDDIT_CLIENT_SECRET = "REDDIT_CLIENT_SECRET"
    REDDIT_CLIENT_ID = "REDDIT_CLIENT_ID"
    REDDIT_REDIRECT_URL = "REDDIT_REDIRECT_URL"

    def __init__(self, google_api_key: str = None, 
                 search_engine_id:     str = None,
                 twitter_bearer_token: str = None,
                 reddit_username:      str = None,
                 reddit_client_secret: str = None,
                 reddit_client_id:     str = None,
                 reddit_redirect_url:  str = None):

        # Googl Config         
        self.google_api_key:       str = google_api_key
        self.search_engine_id:     str = search_engine_id

        # Twitter Config
        self.twitter_bearer_token: str = twitter_bearer_token

        # Reddit Config
        self.reddit_username:      str = reddit_username
        self.reddit_client_secret: str = reddit_client_secret
        self.reddit_client_id:     str = reddit_client_id
        self.reddit_redirect_url:  str = reddit_redirect_url

    # load config data from a file
    @classmethod
    def load_from(cls, filename: str):

        # function to filter comments in lines of the config file
        def cut_comment(line: str) -> str:
            position = line.find(" #")
            return line if position < 0 else line[:position]

        # extract key-value couples from config file
        with open(filename, "r") as file:
            key_value_pairs = [cut_comment(line.strip()).split("=") for line in file.readlines() if len(line.strip()) >= 3 and line.strip()[0] != "#"]
            config = defaultdict(lambda: None, {key_value_pair[0]: key_value_pair[1] for key_value_pair in key_value_pairs if len(key_value_pair) == 2})

        # create config object
        return Configuration(google_api_key       = config[cls.GOOGLE_API_KEY], 
                             search_engine_id     = config[cls.SEARCH_ENGINE_ID],
                             twitter_bearer_token = config[cls.TWITTER_BEARER_TOKEN],
                             reddit_username      = config[cls.REDDIT_USERNAME],
                             reddit_client_id     = config[cls.REDDIT_CLIENT_ID],
                             reddit_client_secret = config[cls.REDDIT_CLIENT_SECRET],
                             reddit_redirect_url  = config[cls.REDDIT_REDIRECT_URL])

    # generate a config file from a number of keys
    @classmethod
    def generate_config_file(cls, filename: str,
                             google_api_key:       str = None, 
                             search_engine_id:     str = None,
                             twitter_bearer_token: str = None,
                             reddit_username:      str = None,
                             reddit_client_secret: str = None,
                             reddit_client_id:     str = None,
                             reddit_redirect_url:  str = None) -> None:

        with open(filename, "w") as file:
            def write_line(key: str, value: str) -> None:
                file.write(f"{'' if value else '# '}{key}={value if value else '???'}\n")

            # Google
            file.write("# Google Config:\n")
            write_line(cls.GOOGLE_API_KEY, google_api_key)
            write_line(cls.SEARCH_ENGINE_ID, search_engine_id)

            # Twitter
            file.write("\n# Twitter Config:\n")
            write_line(cls.TWITTER_BEARER_TOKEN, twitter_bearer_token)

            # Reddit
            file.write("\n# Reddit Config:\n")
            write_line(cls.REDDIT_USERNAME, reddit_username)
            write_line(cls.REDDIT_CLIENT_ID, reddit_client_id)
            write_line(cls.REDDIT_CLIENT_SECRET, reddit_client_secret)
            write_line(cls.REDDIT_REDIRECT_URL, reddit_redirect_url)

    # generate a template config file the user can fill in on their own
    @classmethod
    def generate_template(cls, filename: str) -> None:
        cls.generate_config_file(filename,
                                 google_api_key       = "???",
                                 search_engine_id     = "???",
                                 twitter_bearer_token = "???",
                                 reddit_username      = "???",
                                 reddit_client_id     = "???",
                                 reddit_client_secret = "???",
                                 reddit_redirect_url  = "???")

    # write an existing config object to a file
    def to_file(self, filename: str) -> None:
        self.generate_config_file(filename,
                                  google_api_key       = self.google_api_key,
                                  search_engine_id     = self.search_engine_id,
                                  twitter_bearer_token = self.twitter_bearer_token,
                                  reddit_username      = self.reddit_username,
                                  reddit_client_id     = self.reddit_client_id,
                                  reddit_client_secret = self.reddit_client_secret,
                                  reddit_redirect_url  = self.reddit_redirect_url)
