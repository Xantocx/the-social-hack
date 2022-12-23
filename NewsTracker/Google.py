from NewsTracker import Configuration

from googleapiclient.discovery import build
from typing import List
from math import ceil


# a class wrapping Google' custom search API to perform Google searches in Python code
class GoogleSearch:

    # a container class for search results, providing the title and the URL of a found page
    class SearchResult:

        def __init__(self, json: str) -> None:
            self.title = json["title"]
            self.url = json["link"]

        # parse a list of JSON results provided by the Google API to a list of search results
        @classmethod
        def parse(cls, jsonList: List[str]):
            return [GoogleSearch.SearchResult(json) for json in jsonList]

        # convenient string representation for printing
        def __repr__(self) -> str:
            return f"Google Search Result\n\tTitle: {self.title}\n\tURL: {self.url}"

    def __init__(self, config: Configuration, site: str = None) -> None:
        self.config = config     # config containing API keys
        self.search_site = site  # the page we want to search results on (e.g., only results found on Twitter)

        # create custom search engine object
        self.service = build("customsearch", "v1", developerKey=config.google_api_key)
        self.search_engine = self.service.cse()

    # accessor for search engine id
    @property
    def search_engine_id(self) -> str:
        return self.config.search_engine_id

    # generates a search modifier that is needed when we want to search one specific page only
    @property
    def search_modifier(self) -> str:
        return "" if self.search_site is None else f"site:{self.search_site}"

    # perform search
    # as Google only provides up to 10 results in a single search, we have to break the search in pages to recieve up to 100 results per search
    def search(self, search_term: str, **kwargs) -> List[SearchResult]:

        # setup search iterations
        requested_results = kwargs["num"] if kwargs["num"] else 10
        required_calls = 1

        # calculate number of required calls
        if requested_results > 100:
            raise NotImplementedError("Google does not allow more than 100 results to be delivered.")
        elif requested_results > 10:
            kwargs["num"] = 10
            required_calls = ceil(requested_results / 10)

        # initial setup
        results = []
        kwargs["start"] = 1

        # perform several search calls as necessary
        while required_calls > 0:
            # search for page
            search_term = f"{search_term} {self.search_modifier}"
            current_page = self.search_engine.list(q=search_term, cx=self.search_engine_id, **kwargs).execute()
            results += GoogleSearchResult.parse(current_page["items"])

            # find next page of results
            required_calls -= 1
            kwargs["start"] += 10
            kwargs["num"] = min(requested_results - kwargs["start"] + 1, 10)

        return results

# typealias for convenient class access
GoogleSearchResult = GoogleSearch.SearchResult
