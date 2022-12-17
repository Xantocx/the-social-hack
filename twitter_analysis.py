from NewsTracker.Twitter import *

config = Configuration.load_from(".env")

twitter = TwitterAnalyzer(config)
parents, all = twitter.store_tweets_for_url("https://www.nytimes.com/2022/12/15/business/china-zero-covid-apology.html", "./demo_set.json", 1000, 1)
# parents, all = twitter.load_tweets_from_file("./test.json")
twitter.tweets_to_csv(all, "./demo_set.csv")