from NewsTracker.Config import Configuration

import nltk

# download language models used in our analysis
nltk.download(["vader_lexicon",
               "stopwords", 
               "punkt"], 
               quiet=True)