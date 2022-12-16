from NewsTracker.Config import Configuration
from NewsTracker.Google import GoogleSearch
from NewsTracker.Reddit import RedditAnalyzer


config = Configuration.load_from(".env")
reddit_search = GoogleSearch.create_from(config, "reddit.com")
reddit_analyzer = RedditAnalyzer.create_from(config)

# error whenever >10 search results are requested
results = reddit_search.search("ukraine war", num=10)


group_subs = []
submissions = []

for result in results:
    group_subs.append(reddit_analyzer.get_submissions_from_URL(result.url))

for sub in group_subs:
    for s in sub:
        submissions.append(s)
    if len(sub)<1:
        submissions.remove(s)
    pass


#reddit_analyzer.write_to_csv_stats(subs)

pol_results = reddit_analyzer.get_polarity_results(submissions)
reddit_analyzer.df_from_records(pol_results, to_csv=False, plot=True)