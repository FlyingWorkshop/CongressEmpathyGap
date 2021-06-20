from utils.party import Party
from nltk import word_tokenize
from bs4 import BeautifulSoup
from utils import util
import os
import requests


class Document:
    def __init__(self, label: Party):
        self.label = label
        self.tokens = None

    def display(self):
        print(f"Label: {self.label.name}")
        print(f"Tokens: {util.top_n(self.tokens, 10)}")


class WikiBio(Document):
    ERROR_MSGS = ["Other reasons this message may be displayed:", "may refer to:"]

    def __init__(self, name: str, party: Party):
        super().__init__(label=party)
        self.name = name
        self.url = "https://en.wikipedia.org/wiki/" + util.reformat(self.name)
        self.tokens_cache = "data/cache/" + util.reformat(self.name) + ".csv"
        self.tokens = list(filter(lambda s: s.isalpha(), self._get_tokens()))

    def _get_tokens(self):
        if not os.path.exists(self.tokens_cache):
            text = self._fetch_live_text()
            tokens = word_tokenize(text)
            util.list2csv(tokens, self.tokens_cache)
        else:
            tokens = util.csv2list(self.tokens_cache)
        return tokens

    def _fetch_live_text(self):
        print(f"Fetching data for {self.name}...")
        r = requests.get(self.url)
        soup = BeautifulSoup(r.content, "html.parser")
        text = ' '.join([p.text.strip() for p in soup.find_all('p') if not p.text.isspace()])
        if any(error in text for error in WikiBio.ERROR_MSGS):
            text = ''
        return text

    def display(self):
        super().display()
        print(f"Name: {self.name}")
        print(f"URL: {self.url}")
        print(f"Tokens Cache: {self.tokens_cache}")


class GovWikiBio(WikiBio):
    def __init__(self, name: str, party: Party, term: tuple[int], state: str):
        super().__init__(name, party)
        self.term = term
        self.state = state


class CongWikiBio(WikiBio):
    def __init__(self, name: str, party: Party, twitter: str, dw_nominate: float, votes_with_party: float):
        super().__init__(name, party)
        self.twitter = twitter
        self.dw = dw_nominate
        self.loyalty = votes_with_party
