import os
import requests
from xml.etree.ElementTree import parse


class Bill:
    def __init__(self, legis_num: str):
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


if __name__ == '__main__':
    b = Bill("116s109")
    b._debug()
