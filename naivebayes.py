"""
NOTE: Much of this code structure is directly inspired from an assignment in CS 124 at Stanford University taught by
the amazing Prof. Dan Jurafsky.
"""
from util import WikipediaBiography, Party, ClassData
from nltk import SnowballStemmer, word_tokenize
from nltk.corpus import stopwords
from operator import itemgetter
import numpy as np
import os
import csv
import re


# TODO: implement bigrams
# TODO: train off wikis for senators/house reps and use their DW-nominate as a weighted average


# TODO: use special tweet tokenizer!
class NaiveBayesClassifier:
    TRAINING_FILE = "data/united_states_governors_1775_2020.csv"
    DATES = (1990, 2030)
    STEMMER = SnowballStemmer("english")

    def __init__(self, use_third_parties=False):
        self.stop_words = set(stopwords.words("english") + [WikipediaBiography.FILLER])
        self.data = self.data = [ClassData(par) for par in Party]
        if not use_third_parties:
            self.data.pop()  # removes third party ClassData() object from self.data
            self.data = [ClassData(p) for p in Party if p != Party.OTHER]

        self.training_corpus = []
        self.get_training_corpus(use_third_parties)

        self.train()
        self.classify([])

        # self.display_accuracy
        self.exploration()

    def get_training_corpus(self, use_third_parties: bool) -> None:
        print("### Compiling corpus. This may take awhile...")
        print(f"DATE RANGE: {NaiveBayesClassifier.DATES}")
        with open(NaiveBayesClassifier.TRAINING_FILE) as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # skip header
            seen = set()
            for row in reader:
                name, party = row[0], row[3]
                if name in seen:
                    continue
                dates = tuple(map(int, row[2].split(" - ")))
                if dates[0] < NaiveBayesClassifier.DATES[0] or dates[1] > NaiveBayesClassifier.DATES[1]:
                    continue
                filename = "data/cache/" + name.replace(" ", "_") + ".txt"
                text = None
                if os.path.exists(filename):
                    with open(filename) as f:
                        text = list(f)[0]
                if party == "Republican":
                    party = Party.REPUBLICAN
                elif party == "Democrat":
                    party = Party.DEMOCRAT
                elif use_third_parties:
                    party = Party.OTHER
                else:
                    continue
                self.training_corpus.append(WikipediaBiography(name, party, text))
                seen.add(name)
        print("### Finished compiling corpus!")

    @staticmethod
    def _preprocess(s: str, stop_words: set) -> list[str]:
        s = re.sub(re.compile(r"\[\d*]"), "", s)  # remove footnotes, e.g. "This is a fact. [1]"
        stems = [NaiveBayesClassifier.STEMMER.stem(token) for token in word_tokenize(s) if token.isalpha()]
        return [word for word in stems if word not in stop_words]

    def train(self):
        print("### Training model. This should only take a few seconds...")
        # calculate raw ngram counts and the number of docs in each party
        for doc in self.training_corpus:
            cd = self.data[doc.party.value]
            cd.prior += 1
            ngrams = self._preprocess(doc.text, self.stop_words)
            for ngram in ngrams:
                if ngram not in cd.counts:
                    cd.counts[ngram] = 0
                cd.counts[ngram] += 1

        # normalize priors
        v = len(self.training_corpus)
        for cd in self.data:
            cd.prior /= v

        # merge class vocabularies
        corpus_vocab = set()
        for cd in self.data:
            corpus_vocab.update(cd.counts)

        # calculate likelihoods
        for ngram in corpus_vocab:
            for cd in self.data:
                if ngram in cd.counts:
                    cd.likelihoods[ngram] = (cd.counts[ngram] + 1) / (v + len(cd.counts))
                else:
                    cd.likelihoods[ngram] = 1 / (v + len(cd.counts))
        print("### Finished training model!")

    def classify(self, dev_corpus: list[WikipediaBiography]):
        print("### Classifying developer corpus...")
        guesses = []
        for doc in dev_corpus:
            probabilities = []
            for cd in self.data:
                prob = np.log(cd.prior)
                for ngram in self._preprocess(doc.text, self.stop_words):
                    if ngram in cd.likelihoods:
                        prob += np.log(cd.likelihoods[ngram])
                probabilities.append((prob, cd.party))
            guesses.append(max(probabilities)[1])
        print("### Finished classifying!")
        return guesses

    @staticmethod
    def display_accuracy(guesses: list[Party], documents: list[WikipediaBiography]):
        correct, incorrect = [], []
        for guess, doc in zip(guesses, documents):
            if guess == doc.party:
                correct.append(doc)
            else:
                incorrect.append(doc)
        num_correct, num_incorrect = len(correct), len(incorrect)
        print(f"Model made predictions for {len(documents)} documents.")
        print(f"CORRECT: {num_correct}")
        print(f"INCORRECT: {num_incorrect}")
        print(f"Model Accuracy: {num_correct / num_incorrect}")

    def exploration(self):
        print("### Exploration Mode:")
        while True:
            inp = input("Please type a word ('q' to exit): ")
            if inp == 'q':
                print("Exiting exploration mode.")
                break
            res = self._preprocess(inp)
            if res:
                inp = res[0]
            for cd in self.data:
                print("\t" + str(cd.party) + ":")
                if inp in cd.likelihoods:
                    print(f"\t {cd.likelihoods[inp]=}")
                else:
                    print("\t Not in likelihoods.")
                if inp in cd.counts:
                    print(f"\t {cd.counts[inp]=}")
                else:
                    print("\t Not in counts.")
            if inp in self.stop_words:
                print(f"\t '{inp}' is a stop word!")
            else:
                print(f"\t '{inp}' is NOT a stop word!")
