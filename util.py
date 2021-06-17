import requests
from datetime import date
from bs4 import BeautifulSoup
from enum import Enum
import operator


class Party(Enum):
    DEMOCRAT = 0
    REPUBLICAN = 1
    OTHER = 2


class MiniDict:
    def __init__(self, name: str):
        self.name = name
        self.counts = {}
        self.likelihoods = {}
        self.default_likelihood = 0

    def top_ngrams(self, n=0):
        result = sorted(self.counts.items(), key=operator.itemgetter(1), reverse=True)
        if n == 0 or n > len(self.counts):
            return result
        else:
            return result[:n]


class Corpus:
    def __init__(self, party: Party, use_bigrams: bool):
        self.party = party
        self.size = 0
        self.prior = 0
        self.unigrams = MiniDict("unigrams")
        self.bigrams = None
        if use_bigrams:
            self.bigrams = MiniDict("bigrams")

    def display(self):
        print(f"\t PARTY: {self.party.name}")
        print(f"\t SIZE: {self.size}")
        print(f"\t PRIOR: {self.prior}")
        print(f"\t UNIGRAM TOP COUNTS: {self.unigrams.top_ngrams(5)}")
        if self.bigrams:
            print(f"\t BIGRAM TOP COUNTS: {self.unigrams.top_ngrams(5)}")



class WikiBio:
    ERROR_MESSAGES = ["Other reasons this message may be displayed:", "may refer to:"]
    ERROR_TOKEN = "<UKN>"

    def __init__(self, name: str, party: Party, text=None):
        self.name = name
        self.party = party
        self.url = "https://en.wikipedia.org/wiki/" + self.name.replace(" ", "_")
        self.date_accessed = None
        if text:
            self.text = text
        else:
            self.text = self.get_text()  # loads text if cached; caches and loads text from Wikipedia otherwise

    def get_text(self):
        print(f"Fetching data for {self.name}...")
        today = date.today()
        self.date_accessed = today.strftime("%b-%d-%Y")  # specifies datetime string format, ex: (Jan-01-2001)
        r = requests.get(self.url)
        soup = BeautifulSoup(r.content, "html.parser")
        text = ' '.join([p.text.strip() for p in soup.find_all('p') if not p.text.isspace()])
        if any(error in text for error in WikiBio.ERROR_MESSAGES):
            text = WikiBio.ERROR_TOKEN
        filename = "data/cache/" + self.name.replace(" ", "_") + ".txt"
        with open(filename, 'w') as f:
            f.write(text)
        return text


# TODO: add female republican, male republican; male democrat, female democrat as categories
class CongWikiBio(WikiBio):
    def __init__(self, name: str, party: Party, twitter: str, dw_nominate: float, text=None):
        super().__init__(name, party, text)
        self.twitter = twitter
        self.dw = dw_nominate