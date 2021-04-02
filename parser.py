import requests
import os
from xml.etree.ElementTree import parse

PROPUBLICA_API_KEY = "ADlxAe4YWWQTFEF6oVreM9egfS4uMeWbAV6RWRhf"


class Vote:
    def __init__(self, congress, chamber, legis_num):
        self.roll_num = None
        self.vote_url = None
        self.action = None
        self.date = None
        self.session = None
        self.text = None

        file = f"bills/BILLS-{legis_num}.xml"
        if not os.path.exists(file):
            with open(file, "w") as f:
                url = f"https://www.govinfo.gov/bulkdata/BILLSTATUS/{congress}/{chamber}/BILLSTATUS-{legis_num}.xml"
                r = requests.get(url)
                f.write(r.text)
        self._parse(file)

    def _parse(self, file):
        with open(file) as f:
            tree = parse(f)
            child = tree.find("./bill/recordedVotes/recordedVote")
            self.roll_num = child.find("./rollNumber").text
            self.url = child.find("./url").text
            self.action = child.find("./fullActionName").text
            self.chamber = child.find("./chamber").text
            self.congress = child.find("./congress").text
            self.date = child.find("./date").text
            self.session = child.find("./sessionNumber").text

    def debug(self):
        print("### VOTE DEBUG ###")
        print(f"Roll Call: \t {self.roll_num}")
        print(f"Action: \t {self.action}")
        print(f"Chamber: \t {self.chamber}")
        print(f"Congress: \t {self.congress}")
        print(f"Date: \t\t {self.date}")
        print(f"Session: \t {self.session}")
        print(f"Vote URL: \t {self.url}")


class Bill:
    def __init__(self, legis_num: str):
        self.legis_num = legis_num
        self.congress = legis_num[:3]
        self.chamber = ''.join([char for char in legis_num if char.isalpha()])

        self.url = f"https://www.govinfo.gov/bulkdata/BILLSTATUS/{self.congress}/{self.chamber}/BILLSTATUS-{self.legis_num}.xml"
        self.vote = Vote(self.congress, self.chamber, self.legis_num)

        self._get_roll_call_vote()

    def debug(self):
        print("### BILL DEBUG ###")
        print(f"URL: \t\t {self.url}")
        self.vote.debug()

    def _get_roll_call_vote(self):
        url = f"https://api.propublica.org/congress/v1/{self.congress}/{self.vote.chamber}" \
              f"/sessions/{self.vote.session}/votes/{self.vote.roll_num}.json"
        r = requests.get(url, headers={"X-API-Key": PROPUBLICA_API_KEY})
        data = r.content
        with open(f"data/{self.legis_num}.json", "wb") as f:
            f.write(data)




if __name__ == '__main__':
    b = Bill("116s109")