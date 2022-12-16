from NewsTracker.Twitter import *

config = Configuration.load_from(".env")

twitter = TwitterAnalyzer.create_from(config)
twitter.analyze_url("https://www.nytimes.com/2022/12/15/business/china-zero-covid-apology.html")

# results = twitter.search("tesla")
# print("\n\n\n\n\n")
# print(results.columns)