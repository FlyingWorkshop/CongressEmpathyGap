from enum import Enum


class Party(Enum):
    DEMOCRAT = 0
    REPUBLICAN = 1
    OTHER = 2


def identify(party: str):
    party = party.lower()
    if party == 'democrat' or party == 'd':
        return Party.DEMOCRAT
    elif party == 'republican' or party == 'r':
        return Party.REPUBLICAN
    else:
        return Party.OTHER
