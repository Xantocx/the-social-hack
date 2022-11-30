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
    def read_keys(cls, filename: str) -> Tuple[str, str]:
        with open(filename, "r") as file:
            lines = [line.strip().split("=") for line in file.readlines() if len(line.strip()) >= 3]
            config = {line[0]: line[1] for line in lines if len(line) == 2}

        return config["GOOGLE_API_KEY"], config["CUSTOM_SEARCH_ENGINE_ID"]

    @classmethod
    def create_from(cls, filename: str, site: str = None):
        api_key, search_engine_id = cls.read_keys(filename)
        return GoogleSearch(api_key, search_engine_id, site)

    def search(self, search_term: str, **kwargs) -> List[SearchResult]:
        search_term = f"{search_term} {self.search_modifier}"
        results = self.search_engine.list(q=search_term, cx=self.search_engine_id, **kwargs).execute()
        return GoogleSearch.SearchResult.parse(results["items"])

GoogleSearchResult = GoogleSearch.SearchResult


if __name__ == "__main__":
    API_KEY, SE_ID = GoogleSearch.read_keys("./.env")

    reddit_search = GoogleSearch(API_KEY, SE_ID, "reddit.com")
    results = reddit_search.search("qatar2022", num=10)

    for result in results:
        print(result)
