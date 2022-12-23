from NewsTracker import Configuration
from NewsTracker.URLAnalyzer import URLAnalyzer
from NewsTracker.Google import GoogleSearch

import praw
from prawcore.exceptions import NotFound, Redirect
import json
import sys
import shutil
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import preprocessor as p

from typing import List, Tuple, Dict
from enum import Enum
from json import JSONEncoder
from datetime import datetime, timedelta, timezone
from io import StringIO
from math import ceil
from statistics import mean

from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from pandas import DataFrame, concat
from tqdm import tqdm
from pprint import pprint
import RAKE


class RedditAnalyzer:

    class Submission:

        SIA = SIA() # vader sentiment analysis tool

        DATAFRAME_COLUMS = ["Submission ID", 
                            "Subreddit ID", 
                            "Author", 
                            "Date", #created_utc
                            "Text", 
                            "URL",
                            "Origin", 
                            "Score", 
                            "Upvote Ratio", 
                            "Num Comments", 
                            "Engagement Score",
                            "Negative Sentiment Score",
                            "Neutral Sentiment Score",
                            "Positive Sentiment Score",
                            "Compound Sentiment Score"]


        class SubmissionEncoder(JSONEncoder):
            def default(self, obj):
                return obj.__dict__ 

# not in used but no need to remove it
        class Origin(Enum):
            SEARCH = "search"
            COMMENT = "comment"

        def __init__(self, submission_id:       str, 
                           subreddit_id:        str, 
                           username:            str, 
                           date                ,
                           text:                str, 
                           url:                 str,
                           origin:              Origin,
                           score:               int,
                           upvote_ratio:        int, 
                           comment_count:        int,
                           negative_sentiment:  bool = None,
                           neutral_sentiment:   bool = None,
                           positive_sentiment:  bool = None,
                           compound_sentiment:  bool = None) -> None:

            self.submission_id = submission_id
            self.subreddit_id = subreddit_id
            self.username = username
            #self.date = (datetime.fromtimestamp(date)).strftime('%Y-%m-%d %H:%M:%S')
            # when extracting the data, it needs date.timestamp()
            self.date = date.timestamp()
            self.text = text
            self.url = url
            self._origin = origin.value

            # basic stats 
            self.score          = score 
            self.upvote_ratio   = upvote_ratio
            self.comment_count  = comment_count

            # sentiment analysis
            self.negative_sentiment = negative_sentiment
            self.neutral_sentiment  = neutral_sentiment
            self.positive_sentiment = positive_sentiment
            self.compound_sentiment = compound_sentiment

            self.comments  = []

            self.update_sentiment()

        @property
        def mention(self) -> str:
            return f"@{self.username}"

        @property
        def origin(self) -> Origin:
            return SubmissionOrigin(self._origin)

        @property
        def engagement_score(self) -> int:
            return (4* self.score + 3 * self.comment_count + 3 * self.upvote_ratio) / 10

        @property
        def is_positive(self) -> bool:
            if self.compound_sentiment:
                return self.compound_sentiment > 0
            return None

        @property
        def search_date(self) -> str:
            return str(datetime.strptime(self.date, '%Y-%m-%d %H:%M:%S') - timedelta(days=1)).split()[0]

        @property
        def dataframe_row(self) -> List:
            return [
                self.submission_id,
                self.subreddit_id,
                self.username,
                self.date,
                self.text,
                self.url,
                self._origin,
                self.score,
                self.upvote_ratio,
                self.comment_count,
                self.engagement_score,
                self.negative_sentiment,
                self.neutral_sentiment,
                self.positive_sentiment,
                self.compound_sentiment
            ]

        def add_comments(self, comments: List) -> None:
            self.comments += comments

        def update_sentiment(self) -> None:
            polarity = Submission.SIA.polarity_scores(self.text)
            if self.negative_sentiment is None: self.negative_sentiment = polarity["neg"]
            if self.neutral_sentiment  is None: self.neutral_sentiment  = polarity["neu"]
            if self.positive_sentiment is None: self.positive_sentiment = polarity["pos"]
            if self.compound_sentiment is None: self.compound_sentiment = polarity["compound"]

        def __repr__(self) -> str:
            return f"Submission\n\tID: {self.submission_id}\n\tUsername: {self.username}\n\tText: {self.text}\n\tURL: {self.url}\n"

        @classmethod
        def from_dict(cls, sub_dict):

            if "stats" in sub_dict:
                submission = Submission(
                    submission_id       = sub_dict["submission_id"],
                    subreddit_id        = sub_dict["subreddit_id"],
                    username            = sub_dict["username"],
                    date                = sub_dict["date"],
                    text                = sub_dict["text"],
                    url                 = sub_dict["url"],
                    origin              = SubmissionOrigin(sub_dict["_origin"]),
                    score               = sub_dict["stats"]["score"],
                    upvote_ratio        = sub_dict["stats"]["upvote_ratio"],
                    comment_count       = sub_dict["stats"]["comment_count"],
                    negative_sentiment  = sub_dict["stats"]["negative_sentiment"],
                    neutral_sentiment   = sub_dict["stats"]["neutral_sentiment"],
                    positive_sentiment  = sub_dict["stats"]["positive_sentiment"],
                    compound_sentiment  = sub_dict["stats"]["compound_sentiment"],
                )
            else:
                date_time_format = "%Y-%m-%d %H:%M:%S"
                # Convert the time to the UTC timezone
                submission = Submission(
                    submission_id       = sub_dict["submission_id"],
                    subreddit_id        = sub_dict["subreddit_id"],
                    username            = sub_dict["username"],
                    date                = datetime.strptime(sub_dict["date"], date_time_format).astimezone(timezone.utc),
                    text                = sub_dict["text"],
                    url                 = sub_dict["url"],
                    origin              = SubmissionOrigin(sub_dict["_origin"]),
                    score               = sub_dict["score"],
                    upvote_ratio        = sub_dict["upvote_ratio"],
                    comment_count       = sub_dict["comment_count"],
                    negative_sentiment  = sub_dict["negative_sentiment"],
                    neutral_sentiment   = sub_dict["neutral_sentiment"],
                    positive_sentiment  = sub_dict["positive_sentiment"],
                    compound_sentiment  = sub_dict["compound_sentiment"],
                )

            submission.add_comments([Submission.from_dict(reply) for reply in sub_dict["comments"]])

            return submission

        @classmethod
        def parse_praw(cls, subs, origin: Origin):
            submissions = []
            for sub in subs:  
                submission = Submission(
                    submission_id       = sub.id,
                    subreddit_id        = sub.subreddit.id,
                    username            = sub.author.name if sub.author is not None else None,
                    date                = sub.created_utc, 
                    text                = sub.title,
                    url                 = sub.url,
                    origin              = origin,
                    score               = sub.score,
                    upvote_ratio        = sub.upvote_ratio,
                    comment_count       = sub.num_comments
                )
                submissions.append(submission)
            return submissions

# User
    class User:

        DATAFRAME_COLUMS = ["Username", #name
                            "User ID", #ID
                            "Submission Count", 
                            "Comment count",
                            "Score",
                            "Upvote Ratio",
                            "Engagement Score",
                            "Negative Sentiment Score",
                            "Neutral Sentiment Score",
                            "Positive Sentiment Score",
                            "Compound Sentiment Score"]

        def __init__(self, username: str, id: str) -> None:
            self.username  = username
            self.id = id
            self.comments = []
            self.submissions   = []

        @property
        def submission_count(self) -> int:
            return len(self.submissions)

        @property
        def comment_count(self) -> int:
            return len(self.comments)

        @property
        def score(self) -> int:
            return sum(submission.score for submission in self.submissions)

        @property
        def avg_upvote_ratio(self) -> int:
            return sum(submission.upvote_ratio for submission in self.submissions)/len(self.submissions)

        @property
        def engagement_score(self) -> int:
            return (4* self.score + 3 * self.comment_count + 3 * self.avg_upvote_ratio) / 10

        @property
        def negative_sentiment(self) -> float:
            return sum(submission.negative_sentiment for submission in self.submissions) / self.submission_count

        @property
        def neutral_sentiment(self) -> float:
            return sum(submission.neutral_sentiment for submission in self.submissions) / self.submission_count

        @property
        def positive_sentiment(self) -> float:
            return sum(submission.neutral_sentiment for submission in self.submissions) / self.submission_count

        @property
        def compound_sentiment(self) -> float:
            return sum(submission.compound_sentiment for submission in self.submissions) / self.submission_count

        @property
        def dataframe_row(self) -> List:
            return [
                self.username,
                self.id,
                self.submission_count,
                self.comment_count,
                self.score,
                self.avg_upvote_ratio,
                self.engagement_score,
                self.negative_sentiment,
                self.neutral_sentiment,
                self.positive_sentiment,
                self.compound_sentiment
            ]
        @classmethod
        def parse_praw(cls, users_r):
            users = []
            for user in users_r:
                us = RedditUser(
                    submission_id       = user.id,
                    username            = user.name,
                )
                users.append(us)
            return users

        def add_comment(self, comment) -> None:
            self.comments.append(comment)

        def add_submission(self, submission) -> None:
            self.submissions.append(submission)

# the Rest

    def __init__(self, config: Configuration) -> None:
        self.config = config
        self.api = praw.Reddit(user_agent=config.reddit_username, 
                               client_id=config.reddit_client_id,
                               client_secret=config.reddit_client_secret,
                               redirect_url=os.path.join(config.reddit_redirect_url, 'authorize_callback'))

        self.google = GoogleSearch(self.config, "reddit.com")

    def analyze_url(self, url: str, keywords: List[str], foldername: str) -> None:
        folder = foldername if foldername[-1] == "/" else foldername + "/"

        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "url.txt"), "w") as file:
            file.write(url)
        all_submissions, related_comments = self.store_submissions_for_url(url, keywords, os.path.join(folder, "submissions.json"), limit=5)
        if all_submissions is not None:
            self.analyze(all_submissions, folder)

    def analyze_submissions_file(self, submissions_file: str, foldername: str) -> None:

        submissions = self.load_submissions_from_file(submissions_file)
        # make sure submissions json file is in target folder as well
        folder = foldername if foldername[-1] == "/" else foldername + "/"
        source_file = os.path.abspath(submissions_file)
        target_file = os.path.abspath(os.path.join(folder, "submissions.json"))
        if source_file != target_file:
            os.makedirs(folder, exist_ok=True)
            shutil.copy(source_file, target_file)

        self.analyze(submissions, folder)
        
    def analyze(self, all_submissions: List[Submission], foldername: str) -> None:

        def filename(filename: str) -> str: return os.path.join(foldername, filename)

        print("\nStart analysis...\n")

        print("Extracting users...")
        users = self.get_users_from_submissions(all_submissions)

        print("Write metadata...")
        with open(filename("metadata.txt"), "w") as file:
            file.writelines([f"Submission Count: {len(all_submissions)}\n",
                             f"User Count:  {len(users)}\n"])

        print("Generating CSVs...")
        self.submissions_to_csv(all_submissions, filename("submissions.csv"))
        self.users_to_csv(users, filename("users.csv"))
        
        print("Generating graphs...")
        # submission graphs
        # skip the origin graphs
        #self.submissions_by_origin_graph(all_submissions, filename("submissions_by_origin.png"))
        self.submissions_by_sentiment_graph(all_submissions, filename("submissions_by_sentiment.png"))
        #self.sentiment_by_origin_graph(all_submissions, filename("sentiment_by_origin.png"))
        #self.engagement_by_origin_graph(all_submissions, filename("total_engagement_by_origin.png"), filename("avg_engagement_by_origin.png"))
        self.engagement_by_sentiment_submissions_graph(all_submissions, filename("total_engagement_by_sentiment_submissions.png"), filename("avg_engagement_by_sentiment_submissions.png"))

        # user graphs
        self.engagement_by_sentiment_users_graph(users, filename("total_engagement_by_sentiment_users.png"), filename("avg_engagement_by_sentiment_users.png"))
        #self.user_scatter_log_graph(users, filename("followers_scatter_log.png"))
        #self.user_scatter_filtered_graph(users, filename("followers_scatter_filtered.png"))

        print("\nDone with analysis.")

    def average_engagement(self, submissions: List[Submission]) -> None:
        num_submissions = len(list(submissions))
        if num_submissions == 0: return 0 
        return sum(submission.engagement_score for submission in submissions) / num_submissions

    def submissions_by_origin(self, submissions: List[Submission]) -> Tuple[List[str], List[List[Submission]]]:
        origin_order = [origin.value for origin in TweetOrigin]
        return origin_order, [list(filter(lambda submission: submission._origin == origin, submissions)) for origin in origin_order]

    def submissions_by_sentiment(self, submissions: List[Submission]) -> Tuple[List[float], List[List[Submission]]]:
        sentiment_order = [x/10 for x in range(-10, 11)]
        return sentiment_order, [list(filter(lambda submission: x-0.05 <= submission.compound_sentiment < x+0.05, submissions)) for x in sentiment_order]

    def users_by_sentiment(self, users: Dict[str, User]) -> Tuple[List[float], List[List[User]]]:
        sentiment_order = [x/10 for x in range(-10, 11)]
        return sentiment_order, [list(filter(lambda user: x-0.05 <= user.compound_sentiment < x+0.05, users.values())) for x in sentiment_order]

    def submissions_by_origin_graph(self, submissions: List[Submission], filename: str) -> None:
        origins_x = [origin.value for origin in TweetOrigin]
        origins_y = [len(list(filter(lambda submission: submission._origin == origin, submissions))) for origin in origins_x]

        plt.bar(origins_x, origins_y)
        plt.title("Submissions by Origin")
        plt.xlabel('Origin')
        plt.ylabel('Number of Tweets')
        plt.savefig(filename)
        plt.clf()

    def submissions_by_sentiment_graph(self, submissions: List[Submission], filename: str) -> None:
        sentiment_x = [x/10 for x in range(-10, 11)]
        sentiment_y = [len(list(filter(lambda submission: x-0.05 <= submission.compound_sentiment < x+0.05, submissions))) for x in sentiment_x]

        plt.plot(sentiment_x, sentiment_y)
        plt.title("Submissions by Sentiment")
        plt.xlabel('Compound Sentiment')
        plt.ylabel('Number of Submissions')
        plt.savefig(filename)
        plt.clf()

    def sentiment_by_origin_graph(self, submissions: List[Submission], filename: str) -> None:
        origins_x, submissions_by_origin = self.submissions_by_origin(submissions)
        sentiment_y = [sum(submission.compound_sentiment for submission in origin_submissions) / max(len(origin_submissions), 1) for origin_submissions in submissions_by_origin]

        plt.bar(origins_x, sentiment_y)
        plt.title("Sentiment by Origin")
        plt.xlabel('Origin')
        plt.ylabel('Compound Sentiment')
        plt.savefig(filename)
        plt.clf()

    def engagement_by_origin_graph(self, submissions: List[Submission], total_filename: str, avg_filename: str) -> None:
        origin_x, submissions_by_origin = self.submissions_by_origin(submissions)

        total_engagement_y = [sum(submission.engagement_score for submission in origin_submissions) for origin_submissions in submissions_by_origin]
        avg_engagement_y   = [self.average_engagement(origin_submissions) for origin_submissions in submissions_by_origin]

        # based on sentiment, how high is engagement score
        plt.bar(origin_x, total_engagement_y)
        plt.title("Total Engagement by Origin")
        plt.xlabel('Origin')
        plt.ylabel('Total Engagement')
        plt.savefig(total_filename)
        plt.clf()

        # based on sentiment, how high is the average engagement score
        plt.bar(origin_x, avg_engagement_y)
        plt.title("Average Engagement by Origin")
        plt.xlabel('Origin')
        plt.ylabel('Average Engagement')
        plt.savefig(avg_filename)
        plt.clf()

    def engagement_by_sentiment_submissions_graph(self, submissions: List[Submission], total_filename: str, avg_filename: str) -> None:
        sentiment_x, submissions_by_sentiment = self.submissions_by_sentiment(submissions)
        total_engagement_y = [sum(submission.engagement_score for submission in sentiment_submissions) for sentiment_submissions in submissions_by_sentiment]
        avg_engagement_y   = [self.average_engagement(sentiment_submissions) for sentiment_submissions in submissions_by_sentiment]

        # based on sentiment, how high is engagement score
        plt.plot(sentiment_x, total_engagement_y)
        plt.title("Total Submission Engagement by Sentiment")
        plt.xlabel('Compound Sentiment')
        plt.ylabel('Combined Engagement')
        plt.savefig(total_filename)
        plt.clf()

        # based on sentiment, how high is the average engagement score
        plt.plot(sentiment_x, avg_engagement_y)
        plt.title("Average Submission Engagement by Sentiment")
        plt.xlabel('Compound Sentiment')
        plt.ylabel('Average Engagement')
        plt.savefig(avg_filename)
        plt.clf()

    def engagement_by_sentiment_users_graph(self, users: Dict[str, User], total_filename: str, avg_filename: str) -> None:
        user_list = users.values()

        sentiment_x, users_by_sentiment = self.users_by_sentiment(users)
        total_engagement_y = [sum(user.engagement_score for user in sentiment_users) for sentiment_users in users_by_sentiment]
        avg_engagement_y   = [sum(user.engagement_score for user in sentiment_users) / max(len(sentiment_users), 1) for sentiment_users in users_by_sentiment]

        # based on sentiment, how high is engagement score
        plt.plot(sentiment_x, total_engagement_y)
        plt.title("Total User Engagement by Sentiment")
        plt.xlabel('Compound Sentiment')
        plt.ylabel('Combined Engagement')
        plt.savefig(total_filename)
        plt.clf()

        # based on sentiment, how high is the average engagement score
        plt.plot(sentiment_x, avg_engagement_y)
        plt.title("Average User Engagement by Sentiment")
        plt.xlabel('Compound Sentiment')
        plt.ylabel('Average Engagement')
        plt.savefig(avg_filename)
        plt.clf()

    def user_scatter_log_graph(self, users: Dict[str, User], filename: str) -> None:
        user_list = users.values()

        engagmenent_y = [user.engagement_score for user in user_list]

        submission_counts = [user.submission_count for user in user_list]
        score = [user.score for user in user_list]
        max_submissions = 0
        if len(submission_counts) > 0:
            max_submissions = max(submission_counts)     
        area_submissions = [max(1000 * submission_count / max_submissions, 5) for submission_count in submission_counts]

        sentiment_colors = [(max(1.0 - (user.compound_sentiment + 1), 0), max(user.compound_sentiment, 0), 0.0) for user in user_list]
        
        # score by engagement (log)
        plt.scatter(score, engagmenent_y, s=area_submissions, c=sentiment_colors, alpha=0.5)
        plt.xscale("log")
        plt.yscale("log")
        plt.title("Scatter Plot (size = Submission Count, color = Sentiment)")
        plt.xlabel('Follower Count (log)')
        plt.ylabel('Enagement Score (log)')
        plt.savefig(filename)
        plt.clf()

    def user_scatter_filtered_graph(self, users: Dict[str, User], filename: str) -> None:
        user_list = users.values()
        engagement_limit = sorted(user.engagement_score for user in user_list)[int(0.98 * len(user_list))]
        submissions_limit = sorted(user.submission_count for user in user_list)[int(1 * len(user_list) - 1)]
        scores_limit = sorted(user.score for user in user_list)[int(0.95 * len(user_list))]

        user_list = list(filter(lambda user: user.engagement_score <= engagement_limit and
                                             user.submission_count      <= submissions_limit and
                                             user.score        <= scores_limit, users.values()))
        engagmenent_y = [user.engagement_score for user in user_list]

        submission_counts = [user.submission_count for user in user_list]
        score = [user.score for user in user_list]

        max_submissions = max(submission_counts)
        area_submissions = [max(200 * submission_count / max_submissions, 5) for submission_count in submission_counts]

        sentiment_colors = [(max(1.0 - (user.compound_sentiment + 1), 0), max(user.compound_sentiment, 0), 0.0) for user in user_list]

        # score by engagement
        plt.scatter(score, engagmenent_y, s=area_submissions, c=sentiment_colors, alpha=0.5)
        plt.title("Scatter Plot (size = Submission Count, color = Sentiment)")
        plt.xlabel('Follower Count')
        plt.ylabel('Enagement Score')
        plt.savefig(filename)
        plt.clf()

    def store_submissions_for_url(self, url: str, keywords: List[str], filename: str, limit: int = 10, recusion_depth = 0) -> Tuple[List[Submission], List[Submission]]:
        parent_submissions, related_comments = self.get_submissions_for_url(url, keywords, limit, recusion_depth)
        parsed_submissions = Submission.parse_praw(parent_submissions, Submission.Origin.SEARCH)
        self.store_submissions(parsed_submissions, filename)
        return parsed_submissions, related_comments

    def get_submissions_for_url(self, url: str, keywords: List[str], limit: int = 10, recusion_depth = 0) -> Tuple[List[Submission], List[Submission]]:
        return self.get_submissions(url,  keywords, limit, recusion_depth)

    def get_submissions(self, search_term: str,  keywords: List[str], limit: int = 10, recusion_depth = 0, ignore_ids: List[str] = []) -> Tuple[List[Submission], List[Submission]]:

        def filter_submissions(submissions: List[Submission]):
            return [submission for submission in submissions if submission.id not in ignore_ids]

        print(f"Searching submissions using url '{search_term}'...", end=" ")
        parent_submissions = self.get_submissions_from_non_reddit_url(url=search_term)
        print("Done.\n")

        ignore_ids += [submission.id for submission in parent_submissions]

# TODO get MORE SUBMISSIONS from either Google or API.SEARCH
# OPTION 1:
        # for result in google_search:
        #     #print(f"{result.title}: {result.url}")
        #     google_submissions.append(self.get_submissions_from_reddit_url(url=result.url))
        # google_submissions = [sub for subs in google_submissions for sub in subs]
#OPTION 2:
        print(f"Searching submissions using search term '{search_term}'...", end=" ")
        print(keywords)
        search_submissions = self.search(search_term, keywords, limit)
        search_submissions = filter_submissions(search_submissions)
        parent_submissions += search_submissions
        print("Done.\n")

        if len(parent_submissions) == 0:
            print("Could not find any submissions.\n")
            return None, None

# TODO get comments NOT NEEDED for analysis
        # print("Finding related comments...")
        related_comments = []
        # progress = tqdm(parent_submissions)
        # progress.set_description("Overall Progress")
        # for submission in progress:
        #     related_comments += self.get_related_comments(submission, limit, recusion_depth)
        # print(f"Done.\n\nFound {len(related_submissions)} related comments.")

        return parent_submissions, related_comments



    def get_users_from_submissions(self, submissions: List[Submission]) -> Dict[str, User]:
        usernames = list(set([submission.username for submission in submissions]))
        users_r = []

        for username in usernames:
            if username is not None:
                try:
                    user_reddit = self.api.redditor(username)
                    id = user_reddit.id
                    users_r.append(user_reddit)                    
                except (NotFound, AttributeError) as e:
                    print("Error:", e)
                    users_r.append(None)
            else:
                users_r.append(None)
        users = {}
        for user in users_r:
            if user is not None:
                try:
                    id = user.id
                    users[user.name] =  RedditUser(user.name, id)
                except AttributeError as e:
                    print("Error:", e)
                    users[users.name] = None

        for submission in submissions:
            if submission.username is not None:
                try:
                    users[submission.username].add_submission(submission)
                except KeyError:
                    pass
        return users

    def search(self, search_term: str, keywords: List[str],
                     limit: int = 10
                     ) -> List[Submission]:

        results = []
        
        for keyword in keywords:
            # Get submissions from keywords (Reddit API)
            results.append(self.search_submissions_keywords(keyword, limit))
        # Obtain all submissions from keywords in a list
        results_search = [result for sub_list in results for result in sub_list]

        return results_search

# TODO  get_comments NOT NEEDED
    # def get_comments(self, submission: Submission, limit: int = 1000) -> List[Comment]:
    #     comments = []
    #     return comments

# TODO get_related_comments NOT NEEDED
    def get_related_comments(self, submission: Submission, limit: int = 1000, recursion_depth: int = 0, ignore_ids: List[str] = []) -> List[Submission]:

        def filter_submissions(submissions: List[Submission]):
            return [submission for submission in submissions if submission.submission_id not in ignore_ids]

        return [] 
        comments  = filter_submissions(self.get_comments())      

        submission.add_comments(comments)

        ignore_ids += [submission.submission_id for submission in submissions]

        results = []
        if recursion_depth > 0:

            progress = tqdm(submissions, leave=False)
            progress.set_description(f"Recursion Depth {recursion_depth}")

            for recursive_tweet in progress:
                results += self.get_related_comments(recursive_tweet, limit, recursion_depth - 1, ignore_ids)

        return submissions + results
    
    def store_submissions(self, parent_submissions: List[Submission], filename: str) -> None:
        if parent_submissions is None: return
        with open(filename, "w") as file:
            json.dump(parent_submissions, file, cls=SubmissionEncoder)

    def load_submissions_from_file(self, filename: str) -> Tuple[List[Submission], List[Submission]]:
        with open(filename, "r") as file:
            json_list = json.load(file)
        submissions = [Submission.from_dict(submission) for submission in json_list]
        return submissions

    def submissions_to_dataframe(self, submissions: List[Submission]) -> DataFrame:
        data = [submission.dataframe_row for submission in submissions]
        return DataFrame(data, columns=Submission.DATAFRAME_COLUMS)

    def submissions_to_csv(self, submissions: List[Submission], filename: str) -> None:
        df = self.submissions_to_dataframe(submissions)
        df.to_csv(filename)

    def users_to_dataframe(self, users: Dict[str, User]) -> DataFrame:
        data = [user.dataframe_row for user in users.values()]
        return DataFrame(data, columns=RedditUser.DATAFRAME_COLUMS)

    def users_to_csv(self, users: Dict[str, User], filename: str) -> None:
        df = self.users_to_dataframe(users)
        df.to_csv(filename)

    # get different submissions from a given URL on reddit (https://www.theguardian.co.uk...)
    def get_submissions_from_non_reddit_url(self, url) -> List[Submission]:
        obj = self.api.info(url=url)
        subs = []
        for sub in obj:
            subs.append(sub)
        return subs

    # get submission from a reddit URL (https://www.reddit.com...)
    def get_submissions_from_reddit_url(self, url) -> List[Submission]:
        reddit_obj = self.api.info(url=url)
        sub_list = []
        for obj in reddit_obj:
            if isinstance(obj, praw.models.Subreddit):
                pass
            elif isinstance(reddit_obj, praw.models.Submission):
                #print(f'Submission: {obj} is class {obj.__class__}')
                sub_list.append(obj)
            elif isinstance(reddit_obj, praw.models.Comment):
                #print(f'Comment: {obj} is class {obj.__class__}')
                sub_list.append(obj.submission)
            sub_list.append(obj)
            #print(f'Object: {obj} is class {obj.__class__}')
        return sub_list

    # get submission from keywords 
    def search_submissions_keywords(self, keywords, limit) -> List[Submission]:
        submissions = self.api.subreddit("all").search(query=keywords, sort='relevance', syntax='cloudsearch', limit=limit)
        return submissions
    def get_stopwords(self) -> str:
        return ["a", "and", "the", "in", "of", "to", "on", "at", "for", "with",    "by", "from", "as", "into", "like", "through", "after", "over",    "between", "out", "against", "during", "without", "before", "under",    "around", "among", "within", "along", "up", "down", "off", "above",    "below", "behind", "beyond"]

# Global variables
Submission = RedditAnalyzer.Submission
Submissions = List[Submission]
SubmissionOrigin = Submission.Origin
SubmissionEncoder = Submission.SubmissionEncoder
RedditUser = RedditAnalyzer.User
RedditUsers = List[RedditUser]



