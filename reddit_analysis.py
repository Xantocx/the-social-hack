from NewsTracker import Configuration
from NewsTracker.Google import *
from NewsTracker.Reddit import *

config = Configuration.load_from(".env")

reddit_search = GoogleSearch.create_from(config, "reddit.com")
reddit_analyzer = RedditAnalyzer.create_from(config)

results = reddit_search.search("quantum computers", num=10)

for result in results:
    print(result)
    reddit_analyzer.write_stats_to_csv(result.url)
    print()