import urllib.parse
from typing import Set

# a console printer that allows to stage a number of printing operations, and perform them all at once
# we need this in our final execution of the twitter runner to revert prints that are not needed after all
# just convenience, not necessary
class DelayedPrinter:

    def __init__(self) -> None:
        # queue of elements to print
        self.queue = []

    # check if printer is empty
    @property
    def is_empty(self) -> bool:
        return len(self.queue) == 0

    # stage a new element to print later
    def delay(self, text: str) -> None:
        self.queue.append(text)

    # remove the last element to print
    def pop(self) -> None:
        self.queue.pop()

    # prin all staged elements
    def print(self, text: str = None) -> None:
        if text is not None: self.delay(text)
        for elem in self.queue:
            print(elem)
        self.clear()

    # clear the whole queue
    def clear(self) -> None:
        self.queue = []

# read a list of custom stopwords used for the URLAnalyzer
def read_stopwords() -> Set[str]:
    with open("./NewsTracker/custom_stop_words.txt", "r") as file:
        words = [line.strip() for line in file.readlines()]
    return set(words)

# accessor for custom stopwords
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
