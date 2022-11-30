from newsTracker.google import *
from newsTracker.reddit import *

reddit = GoogleSearch.create_from("./.env", "reddit.com")
results = reddit.search("quantum computers", num=10)

for result in results:
    print(result)
    stats(result.url)
    print()