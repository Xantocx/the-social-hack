from newsTracker import Configuration
from newsTracker.google import *
from newsTracker.reddit import *

config = Configuration.load_from(".env")

reddit = GoogleSearch.create_from(config, "reddit.com")
results = reddit.search("quantum computers", num=10)

for result in results:
    print(result)
    stats(result.url)
    print()