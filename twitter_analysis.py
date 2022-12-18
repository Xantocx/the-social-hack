from NewsTracker.Twitter import *

config = Configuration.load_from(".env")

twitter = TwitterAnalyzer(config)
# parents, all = twitter.store_tweets_for_url("https://www.nytimes.com/2022/12/15/business/china-zero-covid-apology.html", "./demo_set.json", 1000, 1)
# twitter.tweets_to_csv(all, "./demo_set.csv")

twitter.analyze_url("url:https://music.apple.com/us/album/faith-in-the-future-deluxe/1640572729", "./sude_tweet/")

# twitter.analyze_tweets_file("./twitter-testing/tweets.json", "./twitter-testing/")