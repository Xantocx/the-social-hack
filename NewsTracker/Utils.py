import urllib.parse
from typing import Set

class DelayedPrinter:

    def __init__(self) -> None:
        self.queue = []

    @property
    def is_empty(self) -> bool:
        return len(self.queue) == 0

    def delay(self, text: str) -> None:
        self.queue.append(text)

    def pop(self) -> None:
        self.queue.pop()

    def print(self, text: str = None) -> None:
        if text is not None: self.delay(text)
        for elem in self.queue:
            print(elem)
        self.clear()

    def clear(self) -> None:
        self.queue = []

def read_stopwords() -> Set[str]:
    with open("./NewsTracker/custom_stop_words.txt", "r") as file:
        words = [line.strip() for line in file.readlines()]
    return set(words)

custom_stopwords = read_stopwords()



def read_file_and_create_list(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
    return [line.strip() for line in lines]

def get_keywords_from_url(url):
    parsed_url = urllib.parse.urlparse(url)
    path = parsed_url.path
    # Split the path into segments and filter out empty segments
    path_segments = list(filter(bool, path.split("/")))
    path_segments = [list(path.split("-")) for path in path_segments]
    # Return the last segment of the path as the keyword
    if path_segments:
        return path_segments[-1]
    return ""
