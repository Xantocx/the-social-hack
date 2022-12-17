from NewsTracker import Configuration

from googleapiclient.discovery import build
from typing import List
from math import ceil


class GoogleSearch:

    class SearchResult:

        def __init__(self, json: str) -> None:
            self.title = json["title"]
            self.url = json["link"]

        @classmethod
        def parse(cls, jsonList: List[str]):
            return [GoogleSearch.SearchResult(json) for json in jsonList]

        def __repr__(self) -> str:
            return f"Google Search Result\n\tTitle: {self.title}\n\tURL: {self.url}"

    def __init__(self, config: Configuration, site: str = None) -> None:
        self.config = config
        self.search_site = site

        self.service = build("customsearch", "v1", developerKey=config.google_api_key)
        self.search_engine = self.service.cse()

    @property
    def search_engine_id(self) -> str:
        return self.config.search_engine_id

    @property
    def search_modifier(self) -> str:
        return "" if self.search_site is None else f"site:{self.search_site}"

    def search(self, search_term: str, **kwargs) -> List[SearchResult]:

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

        while required_calls > 0:
            # search
            search_term = f"{search_term} {self.search_modifier}"
            current_page = self.search_engine.list(q=search_term, cx=self.search_engine_id, **kwargs).execute()
            results += GoogleSearchResult.parse(current_page["items"])

            # find next page of results
            required_calls -= 1
            kwargs["start"] += 10
            kwargs["num"] = min(requested_results - kwargs["start"] + 1, 10)

        return results

GoogleSearchResult = GoogleSearch.SearchResult
