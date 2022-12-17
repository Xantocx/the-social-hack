from NewsTracker.Twitter import *

config = Configuration.load_from(".env")

twitter = TwitterAnalyzer(config)
tweets = twitter.store_tweets_for_url("https://www.nytimes.com/2022/12/15/business/china-zero-covid-apology.html", "./test.json", 1000, 0)
# tweets = twitter.load_tweets_from_file("./test.json")
twitter.tweets_to_csv(tweets, "./test.csv")