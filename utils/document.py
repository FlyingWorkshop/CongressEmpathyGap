from utils.party import Party
from nltk import word_tokenize
from bs4 import BeautifulSoup
from utils import util
import os
import requests


class Document:
    def __init__(self, label: Party, tokens: list):
        self.label = label
        self.tokens = tokens


class WikiBio(Document):
    ERROR_MSGS = ["Other reasons this message may be displayed:", "may refer to:"]

    def __init__(self, name: str, party: Party):
        super().__init__(label=party, tokens=[])
        self.name = name
        self.url = "https://en.wikipedia.org/wiki/" + util.reformat(self.name)
        self.tokens_cache = "data/cache/" + util.reformat(self.name) + ".csv"
        self.date_accessed = None
        self.tokens = self._get_tokens()

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
        self.date_accessed = util.mark_time()
        r = requests.get(self.url)
        soup = BeautifulSoup(r.content, "html.parser")
        text = ' '.join([p.text.strip() for p in soup.find_all('p') if not p.text.isspace()])
        if any(error in text for error in WikiBio.ERROR_MSGS):
            text = ''
        return text


class GovWikiBio(WikiBio):
    def __init__(self, name: str, party: Party, term: tuple[int], state: str):
        super().__init__(name, party)
        self.term = term
        self.state = state


class CongWikiBio(WikiBio):
    def __init__(self, name: str, party: Party, twitter: str, dw_nominate: float):
        super().__init__(name, party)
        self.twitter = twitter
        self.dw = dw_nominate
