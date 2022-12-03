from newsTracker import Configuration

from googleapiclient.discovery import build
from typing import List, Tuple

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

    def __init__(self, api_key: str, search_engine_id: str, site: str = None) -> None:
        self.api_key: str = api_key
        self.search_engine_id: str = search_engine_id
        self.search_site: str = site

        self.service = build("customsearch", "v1", developerKey=api_key)
        self.search_engine = self.service.cse()

    @property
    def search_modifier(self) -> str:
        return "" if self.search_site is None else f"site:{self.search_site}"

    @classmethod
    def create_from(cls, config: Configuration, site: str = None):
        return GoogleSearch(config.google_api_key, config.search_engine_id, site)

    def search(self, search_term: str, **kwargs) -> List[SearchResult]:
        search_term = f"{search_term} {self.search_modifier}"
        results = self.search_engine.list(q=search_term, cx=self.search_engine_id, **kwargs).execute()
        return GoogleSearch.SearchResult.parse(results["items"])

GoogleSearchResult = GoogleSearch.SearchResult
