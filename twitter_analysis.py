from newsTracker.twitter import *

config = Configuration.load_from(".env")

twitter = TwitterAnalyzer.create_from(config)
twitter.search("tesla")