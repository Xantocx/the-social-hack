from NewsTracker.Twitter import *
from NewsTracker.Utils import DelayedPrinter
import os

# load config
config = Configuration.load_from(".env")
# create twitter analyzer object
twitter = TwitterAnalyzer(config)
# create custom printer object for more pretty printing in console
printer = DelayedPrinter()

# defined folder names for results
result_dir = "./final-results/"
merged_dir = "./merged-results/"

# generate a foldername in result folder
def result_folder(foldername: str) -> str: 
    dir = os.path.join(result_dir, foldername)
    return dir if dir[-1] == "/" else dir + "/"

# generate a foldername in merged result folder
def merged_folder(foldername: str) -> str: 
    dir = os.path.join(merged_dir, foldername)
    return dir if dir[-1] == "/" else dir + "/"

# dictionary for all topics and their respective articles
topics = {
    "elon-musk-twitter": {
        "bbc":                 "https://www.bbc.com/news/business-64010202",
        "euronews":            "https://www.euronews.com/next/2022/12/16/twitter-suspends-the-accounts-of-several-journalists-who-wrote-about-elon-musk",
        "cnbc":                "https://www.cnbc.com/2022/10/27/elon-musk-now-in-charge-of-twitter-ceo-and-cfo-have-left-sources-say.html",
        "wall-street-journal": "https://www.wsj.com/livecoverage/twitter-elon-musk-latest-news",
        "financial-times":     "https://www.ft.com/content/bb047c8f-f97d-4e3a-8bbb-50d0494c8c48",
        "guardian":            "https://www.theguardian.com/technology/2022/dec/17/elon-musk-reinstates-twitter-accounts-of-suspended-journalists"
    }, "worldcup": {
        "bbc":        "https://www.bbc.com/sport/football/63932622",
        "euronews-1": "https://www.euronews.com/2022/11/23/the-qatar-world-cup-kicks-off-with-an-extravagant-ceremony",
        "euronews-2": "https://www.euronews.com/2022/12/18/champions-lionel-messi-leads-argentina-to-electrifying-world-cup-victory-over-france",
        "cnbc":       "https://www.cnbc.com/2022/12/15/fubotv-hit-with-cyber-attack-during-world-cup-semifinal-match.html",
        "sky-spots":  "https://www.skysports.com/football/croatia-vs-morocco/report/463027",
        "fox-sports": "https://www.foxsports.com/stories/soccer/world-cup-2022-5-most-memorable-upsets-of-the-tourney",
        "guardian":   "https://www.theguardian.com/football/live/2022/dec/09/netherlands-v-argentina-world-cup-2022-quarter-final-live-score-updates"
    }, "chatgpt": {
        "guardian":          "https://www.theguardian.com/commentisfree/2022/dec/11/chatgpt-is-a-marvel-but-its-ability-to-lie-convincingly-is-its-greatest-danger-to-humankind",
        "bbc":               "https://www.bbc.com/news/technology-63861322",
        "euronews":          "https://www.euronews.com/next/2022/12/14/chatgpt-why-the-human-like-ai-chatbot-suddenly-got-everyone-talking",
        "cnbc":              "https://www.cnbc.com/2022/12/13/chatgpt-is-a-new-ai-chatbot-that-can-answer-questions-and-write-essays.html",
        "science-focus":     "https://www.sciencefocus.com/future-technology/gpt-3/",
        "bleeping-computer": "https://www.bleepingcomputer.com/news/technology/openais-new-chatgpt-bot-10-dangerous-things-its-capable-of/",
    }, "bts-army": {
        "bbc":                  "https://www.bbc.com/news/world-asia-63944860",
        "euronews":             "https://www.euronews.com/culture/2022/10/10/bts-south-korean-military-leader-says-he-wants-k-pop-band-to-do-military-service",
        "cnbc":                 "https://www.cnbc.com/2022/10/17/bts-stars-to-serve-military-duty-in-south-korea.html",
        "all-kpop":             "https://www.allkpop.com/article/2022/12/bts-jin-to-receive-extra-security-for-military-enlistment",
        "korea-joongang-daily": "https://koreajoongangdaily.joins.com/2022/12/13/entertainment/kpop/Korea-Jin-BTS/20221213094055711.html",
        "guardian":             "https://www.theguardian.com/music/2022/dec/15/bts-military-service-kpop-jin-kim-south-korea",
    }, "iran-women-rights": {
        "bbc":             "https://www.bbc.com/news/world-middle-east-62930425",
        "euronews":        "https://www.euronews.com/2022/12/04/iran-disbands-morality-police-amid-two-and-half-months-of-nationwide-protests",
        "cnbc":            "https://www.cnbc.com/2022/12/05/iran-denies-abolition-of-morality-police-as-three-day-strike-begins.html",
        "middle-east-eye": "https://www.middleeasteye.net/news/iran-mahsa-amini-protests-first-execution-carried-out",
        "arab-news":       "https://www.arabnews.com/node/2209306",
        "guardian":        "https://amp.theguardian.com/global-development/2022/dec/08/iranian-forces-shooting-at-faces-and-genitals-of-female-protesters-medics-say"
    }
}

# function to analyze the above articles
def analyze_topics():

    # iterate through all topics
    for topic, articles in topics.items():

        printer.delay(100 * "-")
        printer.delay(f"\nStart processing topic {topic}...")

        # iterate through all articles
        for index, (source, url) in enumerate(articles.items()):

            printer.delay(f"\nProcessing article {index + 1}/{len(articles)} now...\n")

            # analyze article and write result to respective folder
            dir = result_folder(f"{topic}/{source}")
            if not os.path.exists(dir):
                printer.print()
                twitter.analyze_url(url, dir)
            else:
                printer.pop()

        if printer.is_empty:
            print(f"\nFinished processing topic {topic}.\n")
            print(100 * "-")
            # exit()
        else:
            printer.clear()

def merge_topics():
    
    # iterate through all topics
    for topic, articles in topics.items():

        parents = []
        tweets = []

        # iterate through att articles, and find the corresponding folders to read all tweets from them
        # merge all tweets from all articles in one list
        for source, url in articles.items():
            tweets_file = os.path.join(result_folder(f"{topic}/{source}"), "tweets.json")
            if os.path.exists(tweets_file):
                parent_tweets, all_tweets = twitter.load_tweets_from_file(tweets_file)
                parents += parent_tweets
                tweets += all_tweets

        # create merged results folder
        merged_folder_dir = merged_folder(f"{topic}")
        os.makedirs(merged_folder_dir, exist_ok=True)
        
        # analyze the merged tweets
        twitter.store_tweets(parents, os.path.join(merged_folder_dir, "tweets.json"))
        twitter.analyze(parents, tweets, merged_folder_dir)

        
analyze_topics()
merge_topics()
