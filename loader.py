import csv
import json
import os

from utils.party import Party, identify
from utils.document import GovWikiBio, WikiBio, CongWikiBio, Document
from utils.util import outdated


def get_loader(source):
    loaders = {"congress": _load_congress, "governor": _load_governors, "twitter": _load_twitter}
    return loaders[source]


def _load_congress(dirname, use_third_parties=False):
    documents = []
    for fname in os.listdir(dirname):
        with open(os.path.join(dirname, fname)) as f:
            data = json.load(f)
            roster = data["results"][0]["members"]
        for politician in roster:
            name = politician["first_name"] + ' ' + politician["last_name"]
            party = identify(politician["party"])
            if party == Party.OTHER and not use_third_parties:
                continue
            if "votes_with_party_pct" not in politician:
                continue
            documents.append(CongWikiBio(name, party, politician["twitter_account"], politician["dw_nominate"],
                                         float(politician["votes_with_party_pct"])))
    return documents


def _load_governors(file, timespan, use_third_parties=False):
    documents = []
    with open(file) as f:
        reader = csv.reader(f)
        next(reader)
        seen = set()
        for row in reader:
            name = row[0]
            state = row[1]
            term = tuple(map(int, row[2].split(" - ")))
            party = identify(row[3])
            if name in seen or outdated(term, timespan) or (party == Party.OTHER and not use_third_parties):
                continue
            documents.append(GovWikiBio(name, party, term, state))
            seen.add(name)
    return documents


def _load_twitter(file):
    pass
