"""
The goal of rattenfanger is to make a database of members of congress that correlates
personal details to voting tendencies.


The filename comes from the legend of the pied piper who lured children away...creepy inspiration I know,
but it sounds cool. For more info, read the wiki article: https://en.wikipedia.org/wiki/Pied_Piper_of_Hamelin
"""
from bs4 import BeautifulSoup
import requests

from database import Database


def main(legis_num):
    legis_num = "116s311"
    db = Database(legis_num)
    for senator in db.members:
        print(senator)


if __name__ == '__main__':
    legis_num = "116s311"
    main(legis_num)