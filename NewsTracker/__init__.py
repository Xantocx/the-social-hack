from NewsTracker.Config import Configuration

import nltk

nltk.download(["vader_lexicon",
               "stopwords", 
               "punkt"], 
               quiet=True)