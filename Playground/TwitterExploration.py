import twint

# Set the URL to search for
url = "Sidney Poitier"

# Configure twint to search for the specified URL
c = twint.Config()
c.Search = url
c.Limit = 100  # Limit the number of tweets to 100

# Search for tweets
twint.run.Search(c)

# Print information about the tweets
for tweet in twint.output.tweets_list:
    print(f"{tweet.username} posted the following tweet on {tweet.date}:")
    print(tweet.tweet)
    print()