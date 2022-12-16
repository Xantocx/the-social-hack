from NewsTracker.Config import Configuration
from NewsTracker.Google import GoogleSearch
from NewsTracker.Reddit import RedditAnalyzer

config = Configuration.load_from(".env")

reddit_search = GoogleSearch(config, "reddit.com")
reddit_analyzer = RedditAnalyzer(config)

results = reddit_search.search("quantum computers", num=10)


subs = []

for result in results:
    subs.append(reddit_analyzer.get_submissions_from_URL(result.url))


## Playing with NLTK
## -----------------
headlines = []

for sub in subs:
    for s in sub:
        headlines.append(s.title)
        print(s.title)
    if len(sub)<1:
        subs.remove(sub)

    pass

pol_results = []

for line in headlines:
    pol_score = reddit_analyzer.sia.polarity_scores(line)
    pol_score['headline'] = line
    pol_results.append(pol_score)


print(pol_results)

#reddit_analyzer.write_to_csv_stats(subs)