from NewsTracker.Twitter import *
from NewsTracker.Utils import DelayedPrinter
import os

config = Configuration.load_from(".env")
twitter = TwitterAnalyzer(config)
printer = DelayedPrinter()

result_dir = "./results/"
def folder(foldername: str) -> str: 
    dir = os.path.join(result_dir, foldername)
    return dir if dir[-1] == "/" else dir + "/"

topics = {
    "elon-musk-twitter": [
        "https://www.theguardian.com/technology/2022/nov/19/elon-musk-management-style-twitter-tesla-spacex",
        "https://www.theguardian.com/technology/2022/dec/17/elon-musk-reinstates-twitter-accounts-of-suspended-journalists"
    ], "worldcup": [
        "https://www.theguardian.com/football/live/2022/dec/09/netherlands-v-argentina-world-cup-2022-quarter-final-live-score-updates"
    ], "chatgpt": [
        "https://www.theguardian.com/commentisfree/2022/dec/11/chatgpt-is-a-marvel-but-its-ability-to-lie-convincingly-is-its-greatest-danger-to-humankind"
    ], "bts-army": [
        "https://www.theguardian.com/music/2022/dec/15/bts-military-service-kpop-jin-kim-south-korea"
    ], "iran-women-rights": [
        "https://amp.theguardian.com/global-development/2022/dec/08/iranian-forces-shooting-at-faces-and-genitals-of-female-protesters-medics-say"
    ]

}

for topic, articles in topics.items():

    printer.delay(100 * "-")
    printer.delay(f"\nStart processing topic {topic}...")

    for index, url in enumerate(articles):

        printer.delay(f"\nProcessing article {index + 1}/{len(articles)} now...\n")

        dir = folder(f"{topic}/article{index + 1}")
        if not os.path.exists(dir):
            printer.print()
            twitter.analyze_url(url, dir)
        else:
            printer.pop()

    if printer.is_empty:
        print(f"\nFinished processing topic {topic}.\n")
        print(100 * "-")
    else:
        printer.clear()

# parents, all = twitter.store_tweets_for_url("https://www.nytimes.com/2022/12/15/business/china-zero-covid-apology.html", "./demo_set.json", 1000, 1)
# twitter.tweets_to_csv(all, "./demo_set.csv")

# twitter.analyze_url("https://www.nytimes.com/2022/12/15/business/china-zero-covid-apology.html", "./twitter_testing/")
# twitter.analyze_url('"defenceless lyric video"', "./sude_tweet/")

# twitter.analyze_tweets_file("./twitter_testing/tweets.json", "./twitter_testing/")