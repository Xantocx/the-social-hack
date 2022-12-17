from NewsTracker.Twitter import *

config = Configuration.load_from(".env")

twitter = TwitterAnalyzer(config)
twitter.store_tweets_for_url("https://www.nytimes.com/2022/12/15/business/china-zero-covid-apology.html", "./test.json", 1000, 1)
# twitter.load_tweets_from_file("./test.json")

# results = twitter.search("tesla")
# print("\n\n\n\n\n")
# print(results.columns)