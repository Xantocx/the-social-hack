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