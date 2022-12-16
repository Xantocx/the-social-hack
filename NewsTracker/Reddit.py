from NewsTracker import Configuration

import praw
import csv
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from os.path import join
from IPython import display
from pprint import pprint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid', context='talk', palette='Dark2')


##### WorkInProgress ######

class RedditAnalyzer:

    def __init__(self, config: Configuration, submissions = []) -> None:

        self.config = config

        # Reddit API Object
        self.api = praw.Reddit(user_agent=config.reddit_username, 
                               client_id=config.reddit_client_id,
                               client_secret=config.reddit_client_secret,
                               redirect_url=join(config.reddit_redirect_url, 'authorize_callback'))

        self.sia = SIA() # vader sentiment analysis tool
        self.submissions = submissions # the list of all reddits's submissions objects related to the topic
        self.num_comments = 0
        self.avg_upvote_ratio = 0

        self.update_stats()

    # get different submissions from a given URL on reddit (https://www.theguardian.co.uk...)
    def get_submissions_from_non_reddit_url(self, url):
        obj = self.api.info(url=url)
        subs = []
        for sub in obj:
            subs.append(sub)
        return subs

    # get submission from a reddit URL (https://www.reddit.com...)
    def get_submission_from_reddit_url(self, url):
        return self.api.submission(url=url)

    # gets submissions from either Reddit or non-Reddit URLs
    def get_submissions_from_URL(self, url):
        subs = []
        try: # Reddit URL
            subs.append(self.get_submission_from_reddit_url(url=url))            
        except Exception: # Other URL than Reddit
            try:
                list_subs = self.get_submissions_from_non_reddit_url(url=url)
                for sub in list_subs:
                    subs.append(sub)
            except Exception:
                pass
        return subs

    # renamed from "search"
    # write the stats of the provided submissions
    def write_to_csv_stats(self, subs):

        rows = []
        for sub in subs:
            for s in sub:
                rows.append([s.title, s.score, s.upvote_ratio, s.num_comments, s.is_original_content])

        print(rows)

        fields = ["Title", "Score", "Upvote_Ratio", "Num_comments", "Is_original_content?"]
        with open('submissions.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(fields)
            writer.writerows(rows)


    # renamed from "search"
    # search any URL on reddit that, the URL may not point to reddit itself
    def write_to_csv_from_url(self, url):

        obj = self.api.info(url=url)

        rows = []
        for sub in obj:
            rows.append([sub.title, sub.score, sub.upvote_ratio, sub.num_comments, sub.is_original_content])

        print(rows)

        fields = ["Title", "Score", "Upvote_Ratio", "Num_comments", "Is_original_content?"]
        with open('submissions.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(fields)
            writer.writerows(rows)


    # renamed from "stats"
    # get stats for a reddit post, provided by a proper web url to this reddit post
    def write_to_csv_from_reddit_url(self, url):

        sub = self.api.submission(url=url)
        row = [sub.title, sub.score, sub.upvote_ratio, sub.num_comments, sub.is_original_content]

        print(row)
        
        fields = ["Title", "Score", "Upvote_Ratio", "Num_comments", "Is_original_content?"]
        with open('submissions.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(fields)
            writer.writerow(row)

    def update_stats(self):
        if len(self.submissions) > 0:
            self.num_comments = 0
            sum_upvote_ratio = 0
            for sub in self.submissions:
                self.num_comments += sub.num_comments
                sum_upvote_ratio += sub.upvote_ratio
            self.avg_upvote_ratio = sum_upvote_ratio/len(self.submissions)

    def explore(self, topic):
        # explore other submissions on reddit with the submission's headline
        pass


    def get_polarity_results(self, subs):
        headlines = set()
        for s in subs:
            headlines.add(s.title)
            display.clear_output()

        pol_results = []

        for line in headlines:
            pol_score = self.sia.polarity_scores(line)
            pol_score['headline'] = line
            pol_results.append(pol_score)

        pprint(pol_results[:3], width=100)
        return pol_results

    def df_from_records(self, pol_results, to_csv=False, plot=False):
    # WARNING: RN if there's 0 instances of a negative/positive/neutral label it yields an error
    # acquired from https://www.learndatasci.com/tutorials/sentiment-analysis-reddit-headlines-pythons-nltk/ 
        df = pd.DataFrame.from_records(pol_results)
        df.head()

        df['label'] = 0
        df.loc[df['compound'] > 0.2, 'label'] = 1
        df.loc[df['compound'] < -0.2, 'label'] = -1
        df.head()

        df2 = df[['headline', 'label']]

        # Print positive and Negative Headlines
        print("Positive headlines:\n")
        pprint(list(df[df['label'] == 1].headline)[:5], width=200)
        print("\nNegative headlines:\n")
        pprint(list(df[df['label'] == -1].headline)[:5], width=200)
        print(df.label.value_counts())
        print(df.label.value_counts(normalize=True) * 100)

        if to_csv:
            df2.to_csv('reddit_headlines_labels.csv', mode='a', encoding='utf-8', index=False)

        if plot:
            fig, ax = plt.subplots(figsize=(8, 8))

            counts = df.label.value_counts(normalize=True) * 100

            sns.barplot(x=counts.index, y=counts, ax=ax)

            ax.set_xticklabels(['Negative', 'Neutral', 'Positive'])
            ax.set_ylabel("Percentage")

            plt.show()