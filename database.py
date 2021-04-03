from bills import Bill
from votes import Vote


class Database:
    def __init__(self, legis_num: str):
        self.bill = Bill(legis_num)
        self.vote = Vote(self.bill)