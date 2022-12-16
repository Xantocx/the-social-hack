from NewsTracker import Configuration

from googleapiclient.discovery import build
from typing import List


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

        self.service = build("customsearch", "v1", developerKey=self.api_key)
        self.search_engine = self.service.cse()

    @property
    def api_key(self) -> str:
        return self.config.google_api_key

    @property
    def search_engine_id(self) -> str:
        return self.config.search_engine_id

    @property
    def search_modifier(self) -> str:
        return "" if self.search_site is None else f"site:{self.search_site}"

    def search(self, search_term: str, **kwargs) -> List[SearchResult]:
        search_term = f"{search_term} {self.search_modifier}"
        results = self.search_engine.list(q=search_term, cx=self.search_engine_id, **kwargs).execute()
        return GoogleSearch.SearchResult.parse(results["items"])

GoogleSearchResult = GoogleSearch.SearchResult
