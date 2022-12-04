import praw
import csv
from .google import *


##### WorkInProgress ######

class RedditAnalyzer:

    def __init__(self, 
                 username: str,
                 client_id: str,
                 client_secret: str,
                 redirect_url: str,
                 submissions = []) -> None:

        # Reddit API Object
        self.api = praw.Reddit(user_agent=username, 
                               client_id=client_id,
                               client_secret=client_secret,
                               redirect_url=redirect_url + 'authorize_callback')

        self.submissions = submissions # the list of all reddits's submissions objects related to the topic
        self.num_comments = 0
        self.avg_upvote_ratio = 0

        self.update_stats()

    @classmethod
    def create_from(cls, config: Configuration, submissions = []):
        return RedditAnalyzer(config.reddit_username,
                              config.reddit_client_id,
                              config.reddit_client_secret,
                              config.reddit_redirect_url,
                              submissions)

    # get submissions from URL on reddit that is not from reddit
    def get_submissions_from_url(self, url):
        obj = self.api.info(url=url)
        subs = []
        for sub in obj:
            subs.append(sub)
        return subs

    # get submission from URL on reddit
    def get_submission_from_URL_reddit(self, url):
        return self.api.submission(url=url)

    # renamed from "search"
    # search any URL on reddit that, the URL may not point to reddit itself
    def write_url_posts_to_csv(self, url):

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
    def write_stats_to_csv(self, url):

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
