import os
import requests
from xml.etree.ElementTree import parse
import json
from datetime import date
from bs4 import BeautifulSoup
import nltk


PROPUBLICA_API_KEY = "YOURS HERE"


class Document:
    def __init__(self):
        self.subject = None  # ex: Karel Robot
        self.url = None
        self.date_accessed = None
        self.paragraphs = []


# TODO: cache member data so that you can have one member and store data from multiple votes
# TODO: ex: Karel Robot: { votes { legis_num1 : Yay, legis_num2: Nay }
class Member:
    def __init__(self, d: dict):
        self.id = d['member_id']
        self.name = d['name']
        self.party = d['party']
        self.state = d['state']
        self.vote = d['vote_position']
        self.dw_nominate = d['dw_nominate']

    def display(self):
        print(f"Name: \t\t {self.name}")
        print(f"ID: \t\t {self.id}")
        print(f"Party: \t\t {self.party}")
        print(f"State: \t\t {self.state}")
        print(f"Vote: \t\t {self.name}")
        print(f"DW-NOMINATE: \t {self.dw_nominate}")


class Bill:
    def __init__(self, legis_num: str):
        """
        Caches and stores data for one bill. The bill is specified by the legislation number.

        Example "legis_num" -> "116s109"
        116 - 116th Congress
        s - Senate
        109 - bill number

        """
        # title data
        self.legis_num = legis_num
        self.title = None

        # bill data
        self.bill_num = None
        self.create_date = None
        self.update_date = None
        self.origin_chamber = None
        self.bill_type = None
        self.date_introduced = None

        # latest vote data
        self.roll = None  # roll call number
        self.url = None
        self.action = None  # full action name
        self.chamber = None
        self.congress = None
        self.date = None
        self.session = None  # session number

        file = f"bills/{legis_num}.xml"
        if not os.path.exists(file):
            congress, chamber_abrv = self._decode(legis_num)
            url = f"https://www.govinfo.gov/bulkdata/BILLSTATUS/{congress}/{chamber_abrv}/BILLSTATUS-{legis_num}.xml"
            r = requests.get(url)
            with open(file, "wb") as f:
                f.write(r.content)
        self._make_attrs(file)

    @staticmethod
    def _decode(legis_num: str):
        congress = legis_num[:3]
        chamber_abrv = "".join([char for char in legis_num if char.isalpha()])
        return congress, chamber_abrv

    def _make_attrs(self, file):
        tree = parse(file)
        node = tree.find("./bill")

        # title data
        self.title = node.find("./title").text

        # bill data
        self.bill_num = node.find("./billNumber").text
        self.create_date = node.find("./createDate").text
        self.update_date = node.find("./updateDate").text
        self.origin_chamber = node.find("./originChamber").text
        self.bill_type = node.find("./billType").text
        self.date_introduced = node.find("./introducedDate").text

        # latest vote data
        node = node.find("./recordedVotes/recordedVote")
        self.roll = node.find("./rollNumber").text
        self.url = node.find("./url").text
        self.action = node.find("./fullActionName").text
        self.chamber = node.find("./chamber").text
        self.congress = node.find("./congress").text
        self.date = node.find("./date").text
        self.session = node.find("./sessionNumber").text

    def _debug(self):
        for i, item in enumerate(self.__dict__.items()):
            k, v = item
            num = f"[{i}]".ljust(4, " ")
            lhs = k.upper().ljust(20, " ")
            rhs = v
            print(f"{num} {lhs}: {num} {rhs}")


class Database:
    def __init__(self, legis_num, print_debugging=False):
        self.legis_num = legis_num
        self._print_debugging = print_debugging
        if self._print_debugging:
            print("Initializing database object...")

        # stores a bunch of information related to the bill as a Bill object, makes it easy to interface with
        self.bill = Bill(legis_num)

        # get most recent vote data and cache it
        self.get_voting_data()

        # get member vote data
        self.members = []
        self.get_member_vote_data()

        # cache wikipedia data for each member
        self.documents = []
        self.get_wiki_data()

        if self._print_debugging:
            print("Database initializing COMPLETE!")

    def get_voting_data(self):
        if self._print_debugging:
            print("\t Fetching and caching most recent voting data...")
        if not os.path.exists( f"votes/{self.legis_num}.json"):
            url = f"https://api.propublica.org/congress/v1/{self.bill.congress}/{self.bill.chamber}" \
                  f"/sessions/{self.bill.session}/votes/{self.bill.roll}.json"
            r = requests.get(url, headers={"X-API-Key": PROPUBLICA_API_KEY})
            with open( f"votes/{self.legis_num}.json", "wb") as file:
                file.write(r.content)

    def get_member_vote_data(self):
        if self._print_debugging:
            print("\t Fetching and caching member data...")
        self.members = []
        with open(f"votes/{self.legis_num}.json") as file:
            data = json.load(file)
            for member_data in data['results']['votes']['vote']['positions']:
                self.members.append(Member(member_data))

    def get_wiki_data(self):
        # cache wikipedia paragraph text for each member
        if self._print_debugging:
            print("\t Fetching and caching member wikipedia data... \t *NOTE: This may take awhile.")

        today = date.today()
        date_accessed = today.strftime("%b-%d-%Y")  # specifies datetime string format, ex: (Jan-01-2001)

        for member in self.members:
            page = member.name.replace(" ", "_")
            file = f"data/{page}.json"
            if not os.path.exists(file):
                url = f"https://en.wikipedia.org/wiki/{page}"
                r = requests.get(url)
                soup = BeautifulSoup(r.content, "html.parser")

                # create a Document object
                doc = Document()
                doc.subject = member.name
                doc.date_accessed = date_accessed
                doc.url = url
                for paragraph in soup.find_all('p'):
                    text = paragraph.get_text()
                    text = text.strip()
                    if text:
                        doc.paragraphs.append(text)
                self.documents.append(doc)

                # serialize Document instance as a json
                with open(file, "w") as file:
                    json.dump(doc.__dict__, file, indent=4)





