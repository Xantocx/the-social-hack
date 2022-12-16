from typing import Set

def read_stopwords() -> Set[str]:
    with open("./NewsTracker/custom_stop_words.txt", "r") as file:
        words = [line.strip() for line in file.readlines()]
    return set(words)

custom_stopwords = read_stopwords()