import csv
import string
from datetime import date
from utils.party import Party

INV_IDX = {label.value: label for label in Party}
PUNC_TBL = str.maketrans('', '', string.punctuation)


def make_index(documents):
    index = {}
    for doc in documents:
        if doc not in index:
            index[doc] = []
        index[doc].append(doc)
    return index


def safe_get(key, data):
    if key in data:
        return data[key]
    return 0


def index_max(iterable):
    elem = max(iterable)
    return iterable.index(elem)


def reformat(title: str):
    res = title.translate(PUNC_TBL)
    res = res.replace(' ', '_')
    return res


def mark_time():
    today = date.today()
    return today.strftime("%b-%d-%Y")  # specifies datetime string format, ex: (Jan-01-2001)


def list2csv(li: list, fname: str):
    with open(fname, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(li)


def csv2list(fname: str) -> list:
    with open(fname, newline='') as f:
        reader = csv.reader(f)
        return list(reader)[0]


def outdated(term: tuple[int], timespan: tuple[int]):
    return timespan[0] >= term[0] or term[1] >= timespan[1]


def get_label(i: int):
    return INV_IDX[i]


def top_n(iterable, n):
    bound = min(len(iterable), n)
    return iterable[:bound]
