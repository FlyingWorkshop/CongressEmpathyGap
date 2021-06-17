"""
NOTE: Much of this code structure is directly inspired from an assignment in CS 124 at Stanford University taught by
the amazing Prof. Dan Jurafsky.
"""
from util import WikiBio, Party, Corpus, CongWikiBio, MiniDict
from nltk import SnowballStemmer, word_tokenize
from nltk.corpus import stopwords as nltk_stopwords
from nltk.util import bigrams
from typing import Union
import json
import numpy as np
import os
import csv
import re


# TODO: implement bigrams
# TODO: update likelihood training method
# TODO: train off wikis for senators/house reps and use their DW-nominate as a weighted average
# TODO: use special tweet tokenizer!
# TODO: optionalize stop word filtering
# TODO: write own bigram tokenizer in c++
# TODO: change to np arrays
class NaiveBayesClassifier:
    CONGRESSIONAL_FILES = ["data/training_files/v1_116_senate_members.json",
                           "data/training_files/v1_116_house_members.json"]
    GUBERNATORIAL_FILE = "data/training_files/united_states_governors_1775_2020.csv"
    GOVERNOR_TIME_SPAN = (1990, 2100)  # only include governors w/n this period

    def __init__(self,
                 stemmer=SnowballStemmer("english"),
                 use_stopwords=False,
                 use_third_parties=False,
                 use_bigrams=True):
        # store and interpret parameters
        self.stopwords = None
        if use_stopwords:
            self.stopwords = set(nltk_stopwords.words("english") + [WikiBio.ERROR_TOKEN])
        self.data = None
        if use_third_parties:
            self.data = [Corpus(label, use_bigrams) for label in Party]
        else:
            self.data = [Corpus(label, use_bigrams) for label in Party if label != Party.OTHER]

        # train
        training_data = self._load_documents(NaiveBayesClassifier.CONGRESSIONAL_FILES, use_third_parties)
        self.train(training_data, stemmer, use_stopwords, use_bigrams)

        # classify
        developer_data = self._load_documents([NaiveBayesClassifier.GUBERNATORIAL_FILE], use_third_parties)
        predictions = self.classify(developer_data, stemmer, use_bigrams, debug=True)

        # analyze outcomes
        # self.display_accuracy(predictions, developer_data)
        # self.exploration()

    def _load_documents(self, filenames: list[str], use_third_parties: bool) -> list[Union[WikiBio, CongWikiBio]]:
        corpus = []
        for file in filenames:
            print(f"### Loading file '{file}'...")
            if file.endswith(".csv"):
                corpus += self._load_gubernatorial(file, use_third_parties)
            elif file.endswith(".json"):
                corpus += self._load_congressional(file, use_third_parties)
            else:
                raise Exception("Filename must be either '.csv' or '.json'")
            print(f"### Finished loading file!")
        return corpus

    def _load_gubernatorial(self, filename: str, use_third_parties: bool, timespan=GOVERNOR_TIME_SPAN) -> list[WikiBio]:
        documents = []
        with open(filename) as f:
            reader = csv.reader(f)
            next(reader)
            seen = set()
            for row in reader:
                name = row[0]
                if name in seen:
                    continue
                term = tuple(map(int, row[2].split(" - ")))
                if term[0] < timespan[0] or term[1] > timespan[1]:
                    continue
                party = self.identify(row[3])
                if not use_third_parties and party == Party.OTHER:
                    continue
                cachefile = "data/cache/" + name.replace(" ", "_") + ".txt"
                text = self._load_cached_text(cachefile)
                documents.append(WikiBio(name, party, text))
                seen.add(name)
        return documents

    def _load_congressional(self, filename: str, use_third_parties: bool) -> list[CongWikiBio]:
        documents = []
        with open(filename) as f:
            data = json.load(f)
            roster = data["results"][0]["members"]
        for politician in roster:
            name = politician["first_name"] + " " + politician["last_name"]
            party = self.identify(politician["party"])
            if not use_third_parties and party == Party.OTHER:
                continue
            cachefile = "data/cache/" + name.replace(" ", "_") + ".txt"
            text = self._load_cached_text(cachefile)
            doc = CongWikiBio(name, party, politician["twitter_account"], politician["dw_nominate"], text)
            documents.append(doc)
        return documents

    @staticmethod
    def _load_cached_text(filename: str) -> str:
        text = ""
        if os.path.exists(filename):
            with open(filename) as f:
                text = ' '.join(list(f))
        return text

    @staticmethod
    def identify(s: str) -> Party:
        s = s.lower()
        if s == "r" or s == "republican":
            return Party.REPUBLICAN
        elif s == "d" or s == "democrat":
            return Party.DEMOCRAT
        else:
            return Party.OTHER

    @staticmethod
    def _preprocess(s: str, stemmer, use_bigrams: bool, stop_words=None):
        s = re.sub(re.compile(r"\[\d*]"), "", s)  # remove footnotes, e.g. "This is a fact. [1]"
        ngrams = [stemmer.stem(token) for token in word_tokenize(s) if token.isalpha()]
        if stop_words:
            ngrams = [ngram for ngram in ngrams if ngram not in stop_words]
        if use_bigrams:
            ngrams = bigrams(ngrams)
        return ngrams

    def get_corpus(self, party: Party) -> Corpus:
        return self.data[party.value]

    def _count_ngrams(self, d: MiniDict, doc: Union[WikiBio, CongWikiBio], stemmer,
                      use_stopwords: bool, corpora_vocab=None):
        use_bigrams = d.name == "bigrams"
        for ngram in self._preprocess(doc.text, stemmer, use_bigrams):
            if corpora_vocab != None:
                corpora_vocab.add(ngram)
            if ngram not in d.counts:
                d.counts[ngram] = 0
            d.counts[ngram] += 1

    def train(self, documents: Union[list[WikiBio], list[CongWikiBio]], stemmer, use_stopwords: bool,
              use_bigrams: bool) -> None:
        print(f"### Training model...")
        print("### Calculating corpus sizes, ngram counts, and populating training corpora vocab...")
        unigram_vocab = set()
        bigram_vocab = None
        if use_bigrams:
            bigram_vocab = set()
        for doc in documents:
            corpus = self.get_corpus(doc.party)
            corpus.size += 1
            self._count_ngrams(corpus.unigrams, doc, stemmer, use_stopwords, unigram_vocab)
            if use_bigrams:
                self._count_ngrams(corpus.bigrams, doc, stemmer, use_stopwords, bigram_vocab)

        print("### Calculating corpus priors and likelihoods...")
        for corpus in self.data:
            corpus.prior = corpus.size / len(documents)
            if use_bigrams:
                for bigram in bigram_vocab:
                    bigram_count = 1  # add-one smoothing
                    if bigram in corpus.bigrams.counts:
                        bigram_count += corpus.bigrams.counts[bigram]
                    unigram_count = 1
                    unigram = bigram[0]  # note: bigram is tuple, e.g. (unigram1, unigram2)
                    if unigram in corpus.unigrams.counts:
                        unigram_count += 1
                    corpus.bigrams.likelihoods[bigram] = bigram_count / (unigram_count + len(bigram_vocab))
            else:
                unigram_denominator = len(unigram_vocab) + len(corpus.unigrams.counts)
                for unigram in unigram_vocab:
                    if unigram in corpus.unigrams.counts:
                        unigram_count = 1 + corpus.unigrams.counts[unigram]
                        corpus.unigrams.likelihoods[unigram] = unigram_count / unigram_denominator
                corpus.unigrams.default_likelihood = 1 / unigram_denominator
        print("### Finished training!")

    def classify(self, documents: Union[list[WikiBio], list[CongWikiBio]], stemmer, use_bigrams: bool, debug=False) -> list[Party]:
        # debugging vars
        true_dem = 0
        false_dem = 0
        true_rep = 0
        false_rep = 0
        print("### Classifying...")
        predictions = []
        for doc in documents:
            scores = [np.log(corpus.prior) for corpus in self.data]
            for ngram in self._preprocess(doc.text, stemmer, use_bigrams):
                for i, corpus in enumerate(self.data):
                    if use_bigrams and ngram in corpus.bigrams.likelihoods:
                        scores[i] += np.log(corpus.bigrams.likelihoods[ngram])
                        # print(f"{ngram}, {corpus.bigrams.likelihoods[ngram]}")
                    elif not use_bigrams and ngram in corpus.unigrams.likelihoods:
                        scores[i] += np.log(corpus.unigrams.likelihoods[ngram])
                        # print(f"{corpus.unigrams.likelihoods[ngram]=}")
            if all(val == 0 for val in scores):
                continue
            high_score = max(scores)
            i = scores.index(high_score)
            prediction = self.data[i].party
            predictions.append(prediction)
            if debug:
                print(doc.name)
                print(doc.party.name)
                print(scores)
                if scores[0] > scores[1]:
                    if doc.party == Party.DEMOCRAT:
                        print("TRUE DEMOCRAT")
                        true_dem += 1
                    else:
                        print("FALSE DEMOCRAT")
                        false_dem += 1
                else:
                    if doc.party == Party.DEMOCRAT:
                        print("TRUE REPUBLICAN")
                        true_rep += 1
                    else:
                        print("FALSE REPUBLICAN")
                        false_rep += 1
                print("---")
        print("### Finished classifying!")
        if debug:
            print(f"{true_dem=}")
            print(f"{false_dem=}")
            print(f"{true_rep=}")
            print(f"{false_rep=}")
            dem_precision = true_dem / (true_dem + false_dem + 1)
            rep_precision = true_rep / (true_rep + false_rep + 1)
            dem_recall = true_dem / (true_dem + false_rep + 1)
            rep_recall = true_rep / (true_rep + false_dem + 1)
            overall_accuracy = (true_rep + true_dem) / len(documents)
            dem_counts = predictions.count(Party.DEMOCRAT)
            rep_counts = predictions.count(Party.REPUBLICAN)
            print(f"{dem_counts=}")
            print(f"{rep_counts=}")
            print(f"{dem_precision=}")
            print(f"{rep_precision=}")
            print(f"{dem_recall=}")
            print(f"{rep_recall=}")
            print(f"{overall_accuracy=}")
        return predictions

    # @staticmethod
    # def display_accuracy(predictions: list[Party], documents: Union[list[WikiBio], list[CongWikiBio]]):
    #     print("### Checking accuracy...")
    #
    # def exploration(self):
    #     print("### Exploration Mode:")
    #     while True:
    #         inp = input("Please type a word ('q' to exit): ")
    #         if inp == 'q':
    #             print("Exiting exploration mode.")
    #             break
    #         res = self._preprocess(inp, self.stemmer, self.stopwords)
    #         if res:
    #             inp = res[0]
    #         for cd in self.data:
    #             print("\t" + str(cd.party) + ":")
    #             if inp in cd.likelihoods:
    #                 print(f"\t {cd.likelihoods[inp]=}")
    #             else:
    #                 print("\t Not in likelihoods.")
    #             if inp in cd.counts:
    #                 print(f"\t {cd.counts[inp]=}")
    #             else:
    #                 print("\t Not in counts.")
    #         if inp in self.stopwords:
    #             print(f"\t '{inp}' is a stop word!")
    #         else:
    #             print(f"\t '{inp}' is NOT a stop word!")
