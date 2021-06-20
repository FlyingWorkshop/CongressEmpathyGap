"""
A short script to request data from the Propublica Congress API
"""
import requests
import json
import os

KEY = "YOURS_HERE"


def main():
    congress = 114
    chamber = "house"
    filename = f"data/sources/congress/v1_{congress}_{chamber}_members.json"
    if not os.path.exists(filename):
        url = f"https://api.propublica.org/congress/v1/{congress}/{chamber}/members.json"
        headers = {"X-API-Key": KEY}
        r = requests.get(url, headers=headers)
        data = r.json()
        with open(filename, 'w') as outfile:
            json.dump(data, outfile, indent=4)


if __name__ == "__main__":
    main()
