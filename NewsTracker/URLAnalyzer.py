from typing import List
from collections import defaultdict

from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer

from NewsTracker.Utils import custom_stopwords


class URLAnalyzer:

    ENGLISH_STOPWORDS = stopwords.words("english")
    CUSTOM_STOPWORDS = custom_stopwords

    def __init__(self, url: str) -> None:
        self.url = url

        # load webpage data
        self.request = Request(self.url, headers={'User-Agent' : "Magic Browser"})  # Perform request to webpage
        self.html = urlopen(self.request).read()                                    # Open the URL and read the HTML content
        self.soup = BeautifulSoup(self.html, "html.parser")                         # Use BeautifulSoup to parse the HTML content

        # Lazy properties
        self._title = None
        self._text = None
        self._tokens = None
        self._search_terms = None

    @property
    def title(self) -> str:
        if not self._title:
            title = self.soup.title
            self._title = title.string if title else None
        return self._title

    @property
    def text(self) -> str:
        if not self._text:
            self._text = self.soup.get_text()
        return self._text

    @property
    def tokens(self) -> List[str]:
        if not self._tokens:
            self._tokens = [token.lower() for token in word_tokenize(self.text)] 
        return self._tokens

    @property
    def search_terms(self) -> List[str]:

        if not self._search_terms:

            # maybe extract named entities?

            # filter tokens
            filtered_tokens = [token for token in self.tokens if token.isalpha() and token not in self.ENGLISH_STOPWORDS and token not in self.CUSTOM_STOPWORDS]
            filter_text = " ".join(filtered_tokens)

            # Calculate the TF-IDF scores for the keywords
            vectorizer = TfidfVectorizer()
            tfidf_scores = vectorizer.fit_transform([filter_text])  # returns an array for each string element in the input array, containing the scores of the words in that document
            feature_names = vectorizer.get_feature_names_out()

            ranked_tokens = defaultdict(lambda: 0, {token: score for token, score in zip(feature_names, tfidf_scores.toarray()[0])})

            # Generate bi-grams and tri-grams from the filtered tokens
            bi_grams = [" ".join(gram) for gram in ngrams(filtered_tokens, 2)]
            tri_grams = [" ".join(gram) for gram in ngrams(filtered_tokens, 3)]
            grams = bi_grams + tri_grams

            sorted(grams, key=lambda gram: sum(ranked_tokens[token] for token in gram.split()) / len(gram.split()))
            grams.reverse()

            # for gram in grams[:10]:
            #     print(f"{gram}: {sum(ranked_tokens[token] for token in gram.split()) / len(gram.split())}")

            self._search_terms = grams

        return self._search_terms
