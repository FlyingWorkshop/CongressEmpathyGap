from bills import Bill
import os
import requests
import json

PROPUBLICA_API_KEY = "YOURS HERE"


class Member:
    def __init__(self, d: dict):
        self.id = d['member_id']
        self.name = d['name']
        self.party = d['party']
        self.state = d['state']
        self.vote = d['vote_position']
        self.dw_nominate = d['dw_nominate']

    def __str__(self):
        return self.name


class Vote:
    def __init__(self, bill: Bill):
        file = f"votes/{bill.legis_num}.json"
        if not os.path.exists(file):
            url = f"https://api.propublica.org/congress/v1/{bill.congress}/{bill.chamber}/sessions/{bill.session}/votes/{bill.roll}.json"
            r = requests.get(url, headers={"X-API-Key": PROPUBLICA_API_KEY})
            with open(file, "wb") as f:
                f.write(r.content)

        with open(file) as f:
            json_data = json.load(f)  # list of dicts

        temp_data = json_data['results']['votes']['vote']['positions']

        self.members = [Member(d) for d in temp_data]
        self.votes = {}
        for member in self.members:
            if member.vote not in self.votes:
                self.votes[member.vote] = []
            self.votes[member.vote].append(member)
