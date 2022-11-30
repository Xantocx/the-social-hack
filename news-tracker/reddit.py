import praw
import time
import csv

import sys
sys.path.append("./news-tracker")

from reddit import *

#Delete keys before handing in the notebook
r = praw.Reddit(user_agent='eguiwan_kenobi', client_id='udaTQA7LQKtu5VH68BUmng',
                      client_secret='dh-qgDHjSXziTfTgvEadkP-MOk9AAQ',
                      redirect_url='https://www.reddit.com/prefs/apps/'
                                   'authorize_callback')

obj = r.info(url="https://www.bbc.co.uk/news/world-europe-56720589")

titles = []
scores = []
upvote_ratios = []
nums_comments = []
original_contents = []
rows = []
for sub in obj:
    rows.append([sub.title, sub.score, sub.upvote_ratio, sub.num_comments, sub.is_original_content])

print(rows)
fields = ["Title", "Score", "Upvote_Ratio", "Num_comments", "Is_original_content?"]

with open('submissions.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(fields)
    writer.writerows(rows)

