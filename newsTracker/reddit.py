import praw
import csv
from .google import *


##### WorkInProgress ######

class Analyzer:
    def __init__(self, submissions, stats) -> None:
        self.submissions = submissions # the list of all reddits's submissions objects related to the topic
        if len(submissions) > 0:
            num_comments = 0
            sum_upvote_ratio = 0
            for sub in submissions:
                num_comments += sub.num_comments
                sum_upvote_ratio += sub.upvote_ratio
            avg_upvote_ratio = sum_upvote_ratio/len(submissions)
        self.stats = {
            "Total_comments": num_comments,
            "Avg_upvote_ratio": avg_upvote_ratio
            }
    
    # @classmethod
    # def explore(topic):
    # # explore other submissions on reddit with the submission's headline
        

    @classmethod
    def update_stats(self):
        if len(self.submissions) > 0:
            num_comments = 0
            sum_upvote_ratio = 0
            for sub in self.submissions:
                num_comments += sub.num_comments
                sum_upvote_ratio += sub.upvote_ratio
            avg_upvote_ratio = sum_upvote_ratio/len(self.submissions)
        self.stats = {
            "Total_comments": num_comments,
            "Avg_upvote_ratio": avg_upvote_ratio
            }


# Reddit API Object

#Delete keys before handing in the notebook
r = praw.Reddit(user_agent='eguiwan_kenobi', client_id='udaTQA7LQKtu5VH68BUmng',
                      client_secret='dh-qgDHjSXziTfTgvEadkP-MOk9AAQ',
                      redirect_url='https://www.reddit.com/prefs/apps/'
                                   'authorize_callback')



# get submissions from URL on reddit that is not from reddit
def get_submissions_from_url(url):
    obj = r.info(url=url)
    subs = []
    for sub in obj:
        subs.append(sub)
    return subs

# get submission from URL on reddit
def get_submission_from_URL_reddit(url):
    return r.submission(url=url)
    

#######
# CSV
#######
# search any URL on reddit that is not from reddit
def search(url):

    obj = r.info(url=url)

    rows = []
    for sub in obj:
        rows.append([sub.title, sub.score, sub.upvote_ratio, sub.num_comments, sub.is_original_content])

    print(rows)

    fields = ["Title", "Score", "Upvote_Ratio", "Num_comments", "Is_original_content?"]
    with open('submissions.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(fields)
        writer.writerows(rows)


# # get stats for a URL on reddit
def stats(url):

    sub = r.submission(url=url)
    row = [sub.title, sub.score, sub.upvote_ratio, sub.num_comments, sub.is_original_content]

    print(row)
    
    fields = ["Title", "Score", "Upvote_Ratio", "Num_comments", "Is_original_content?"]
    with open('submissions.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(fields)
        writer.writerow(row)
