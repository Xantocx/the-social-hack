from NewsTracker import Configuration
from NewsTracker.Google import GoogleSearch

config = Configuration.load_from(".env")

reddit_search = GoogleSearch(config, "reddit.com")
results = reddit_search.search("tesla", num=100)

print("\n\n")
for result in results:
    print(result)
print()
print(len(results))