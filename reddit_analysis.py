from NewsTracker.Config import Configuration
from NewsTracker.Google import GoogleSearch
from NewsTracker.Reddit import RedditAnalyzer

config = Configuration.load_from(".env")

reddit_search = GoogleSearch.create_from(config, "reddit.com")
reddit_analyzer = RedditAnalyzer.create_from(config)

results = reddit_search.search("quantum computers", num=10)


subs = []

for result in results:
    subs.append(reddit_analyzer.get_submissions_from_URL(result.url))

for sub in subs:
    if len(sub)<1:
        subs.remove(sub)
    pass

reddit_analyzer.write_to_csv_stats(subs)