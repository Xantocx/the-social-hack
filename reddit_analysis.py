from NewsTracker.Reddit import *
from NewsTracker.Utils import DelayedPrinter
import os

config = Configuration.load_from(".env")

reddit = RedditAnalyzer(config)
printer = DelayedPrinter()

result_dir = "./final-results/"
merged_dir = "./merged-results/"

keywords = [["twitter banned journalists", "elon musk bans journalist"], ["UN condems Twitter journalist account", "EU sanctions Twitter journalists"], ["Twitter new CEO Elon Musk", "Twitter new CFO Elon Musk"],
["Elon Musk buys Twitter", "Twitter employees reacts with humour"], ["Musk looks for funds", "Musk seeks to sell Twitter shares"], ["Musk reinstates journalists", "Elon reinstates accounts journalists"],
["", ""], ["", ""], ["", ""], ["", ""], ["", ""], ["", ""], ["", ""],
["", ""], ["", ""], ["", ""], ["", ""], ["", ""], ["", ""], ["", ""], ["", ""], ["", ""], ["", ""], ["", ""], ["", ""], ["", ""], ["", ""], 
["", ""], ["", ""], ["", ""], ["", ""], ["", ""], ["", ""]]


def result_folder(foldername: str) -> str: 
    dir = os.path.join(result_dir, foldername)
    return dir if dir[-1] == "/" else dir + "/"

def merged_folder(foldername: str) -> str: 
    dir = os.path.join(merged_dir, foldername)
    return dir if dir[-1] == "/" else dir + "/"

#parents, comments = reddit.store_submissions_for_url("https://www.nytimes.com/2022/12/15/business/china-zero-covid-apology.html", ["covid china", "china policy covid", "china zero covid"], "./demo_set.json", 1000, 1)

keywords = {
    "elon-musk-twitter": {
        "bbc":                 ["twitter banned journalists", "elon musk bans journalist"],
        "euronews":            ["UN condems Twitter journalist account", "EU sanctions Twitter journalists"],
        "cnbc":                ["Twitter new CEO Elon Musk", "Twitter new CFO Elon Musk"],
        "wall-street-journal": ["Elon Musk buys Twitter", "Twitter employees reacts with humour"],
        "financial-times":     ["Musk looks for funds", "Musk seeks to sell Twitter shares"],
        "guardian":            ["Musk reinstates journalists", "Elon reinstates accounts journalists"],
    }, "worldcup": {
        "bbc":        ["World Cup Argentina Messi", "Argentina wins the final France"],
        "euronews-1": ["Qatar 2022 extravagant ceremony", "Qatar 2022 kicks off"],
        "euronews-2": ["Electrifying victory Argentina World Cup", "Messi Argentina champions World Cup"],
        "cnbc":       ["cyberattack world cup semifinal", "FuboTV cyberattack"],
        "sky-spots":  ["Croatia Morocco third place", "Orsic winner Croatia Morocco"],
        "fox-sports": ["World Cup 2022 memorable upsets", "5 upsets Qatar 2022"],
        "guardian":   ["Argentina Netherlands semifinal", "Argentina wins Netherlands"],
    }, "chatgpt": {
        "guardian":          ["chatGPT danger lie convince", "chatGPT wonder but threat"],
        "bbc":               ["new chatbot chatGPT", "chatGPT new AI"],
        "euronews":          ["everyone talking chatGPT", "chatGPT people talking"],
        "cnbc":              ["chatGPT write essays", "chatGPT answer questions"],
        "science-focus":     ["chatGPT everything you need to know", "chatGPT OpenAI GPT-3 tool"],
        "bleeping-computer": ["10 dangerous things chatGPT", "Open AI dangerous things chatGPT"],
    }, "bts-army": {
        "bbc":                  ["jin bts begins military", "jin bts k-pop military"],
        "euronews":             ["south korea leader BTS military", "south korea wants BTS military"],
        "cnbc":                 ["BTS to serve in south korea"],
        "all-kpop":             ["jin to receive extra security"],
        "korea-joongang-daily": ["jin safety control BTS"],
        "guardian":             ["jin military security BTS"],
    }, "iran-women-rights": {
        "bbc":             ["Iran young woman dies", "morality police arrest Iran"],
        "euronews":        ["Iran disbands morality police", "Iran disbands police after protests"],
        "cnbc":            ["Iran 3 strike days", "Iran morality police 3 day strike"],
        "middle-east-eye": ["Mahsa Amini first execution", "Mahsa first execution Iran"],
        "arab-news":       ["echoes of revolution Iran", "1979 Iran revolution today"],
        "guardian":        ["Iran police shooting genitalia protesters", "Iran police shooting women face"]
    }
}

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

def analyze_topics():
    keywords_index = 0
    for topic, articles in topics.items():
        printer.delay(100 * "-")
        printer.delay(f"\nStart processing topic {topic}...")
        for index, (source, url) in enumerate(articles.items()):
            search_words = keywords[topic][source]
            printer.delay(f"\nProcessing article {index + 1}/{len(articles)} now...\n")

            dir = result_folder(f"{topic}/{source}")
            if not os.path.exists(dir):
                printer.print()
                reddit.analyze_url(url, search_words, dir)
            else:
                printer.pop()
            keywords_index += 1

        if printer.is_empty:
            print(f"\nFinished processing topic {topic}.\n")
            print(100 * "-")
            # exit()
        else:
            printer.clear()

def merge_topics():
    for topic, articles in topics.items():

        all_submissions = []
        submissions = []

        for source, url in articles.items():
            submissions_file = os.path.join(result_folder(f"{topic}/{source}"), "submissions.json")
            if os.path.exists(submissions_file):
                submissions= reddit.load_submissions_from_file(submissions_file)
                all_submissions += submissions
        merged_folder_dir = merged_folder(f"{topic}")
        os.makedirs(merged_folder_dir, exist_ok=True)
        
        reddit.store_submissions(all_submissions, os.path.join(merged_folder_dir, "submissions.json"))
        reddit.analyze(all_submissions, merged_folder_dir)

#analyze_topics()
merge_topics()

# parents, all = reddit.store_submissions_for_url("https://www.nytimes.com/2022/12/15/business/china-zero-covid-apology.html", "./demo_set.json", 1000, 1)
# reddit.submissions_to_csv(all, "./demo_set.csv")

# reddit.analyze_url("https://www.nytimes.com/2022/12/15/business/china-zero-covid-apology.html", "./reddit_testing/")
# reddit.analyze_url('"defenceless lyric video"', "./sude_tweet/")

# reddit.analyze_submissions_file("./reddit_testing/submissions.json", "./reddit_testing/")