import requests
from datetime import date
from bs4 import BeautifulSoup
from enum import Enum


class Party(Enum):
    DEMOCRAT = 0
    REPUBLICAN = 1
    OTHER = 2


class ClassData:
    def __init__(self, party: Party):
        self.party = party
        self.prior = 0
        self.counts = {}
        self.likelihoods = {}


class WikipediaBiography:
    ERROR_MESSAGES = ["Other reasons this message may be displayed:", "may refer to:"]
    FILLER = "<UKN>"

    def __init__(self, name: str, party: Party, text=None):
        self.name = name
        self.party = party
        self.url = "https://en.wikipedia.org/wiki/" + self.name.replace(" ", "_")
        self.date_accessed = None
        if text:
            self.text = text
        else:
            print(self.name)
            print("whoops!")
            self.text = self.get_text()  # loads text if cached; caches and loads text from Wikipedia otherwise

    def get_text(self):
        """
        >>> WikipediaBiography("Bill Walker", Party.DEMOCRAT)
        """
        print(f"Fetching data for {self.name}...")
        today = date.today()
        self.date_accessed = today.strftime("%b-%d-%Y")  # specifies datetime string format, ex: (Jan-01-2001)
        r = requests.get(self.url)
        soup = BeautifulSoup(r.content, "html.parser")
        text = ' '.join([p.text.strip() for p in soup.find_all('p') if not p.text.isspace()])
        print(text)
        if any(error in text for error in WikipediaBiography.ERROR_MESSAGES):
            text = WikipediaBiography.FILLER
        filename = self.name.replace(" ", "_")
        with open("data/cache/" + filename + ".txt", 'w') as f:
            f.write(text)
        return text


