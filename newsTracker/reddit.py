import praw
import csv

#Delete keys before handing in the notebook
r = praw.Reddit(user_agent='eguiwan_kenobi', client_id='udaTQA7LQKtu5VH68BUmng',
                      client_secret='dh-qgDHjSXziTfTgvEadkP-MOk9AAQ',
                      redirect_url='https://www.reddit.com/prefs/apps/'
                                   'authorize_callback')

# "https://www.bbc.co.uk/news/world-europe-56720589"

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


# get stats for a URL on reddit
def stats(url):

    sub = r.submission(url=url)
    row = [sub.title, sub.score, sub.upvote_ratio, sub.num_comments, sub.is_original_content]

    print(row)
    
    fields = ["Title", "Score", "Upvote_Ratio", "Num_comments", "Is_original_content?"]
    with open('submissions.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(fields)
        writer.writerow(row)
