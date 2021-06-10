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

# TODO: find out why de-serialization is so slow by debug create training
# TODO: increase unigram probability by debugging classify and train; fix likelihood!
# TODO: implement bigrams


# TODO: use special tweet tokenizer!
class NaiveBayesClassifier:
    TRAINING_FILE = "data/united_states_governors_1775_2020.csv"
    USE_GOVERNORS_BETWEEN = (1945, 2030)

    def __init__(self, training_ratio=0.7, custom_stop_words=2, include_third_parties=True):
        self.stemmer = SnowballStemmer("english")
        self.stop_words = self.make_stop_words()
        if include_third_parties:
            self.data = [ClassData(p) for p in Party]
        else:
            self.data = [ClassData(p) for p in Party if p != Party.OTHER]

        print("### Compiling corpus. This may take awhile...")
        corpus = self.make_training_corpus(NaiveBayesClassifier.TRAINING_FILE, include_third_parties)
        print("### Finished compiling corpus!")
        b = int(len(corpus) * training_ratio)
        training_corpus = corpus[:b]
        dev_corpus = corpus[b:]

        print("### Training model. This should only take a few seconds...")
        self.train(training_corpus, custom_stop_words)
        print("### Finished training model!")

        guesses = self.classify(dev_corpus)
        self.display_accuracy(guesses, dev_corpus)

        print("### Exploration Mode:")
        self.exploration()

    @staticmethod
    def make_stop_words():
        result = set(stopwords.words("english"))
        result.add(WikipediaBiography.FILLER)
        return result

    @staticmethod
    def make_training_corpus(training_file: str, include_third_parties: bool) -> list[WikipediaBiography]:
        corpus = []
        print("DATE RANGE:")
        print("\t" + str(NaiveBayesClassifier.USE_GOVERNORS_BETWEEN))
        with open(training_file) as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # skip header
            seen = set()
            for row in reader:
                name = row[0]
                if name in seen:
                    continue
                time_in_office = tuple(map(int, row[2].split(" - ")))
                if (time_in_office[0] < NaiveBayesClassifier.USE_GOVERNORS_BETWEEN[0] or
                        time_in_office[1] > NaiveBayesClassifier.USE_GOVERNORS_BETWEEN[1]):
                    continue
                if row[3] == "Democrat":
                    party = Party.DEMOCRAT
                elif row[3] == "Republican":
                    party = Party.REPUBLICAN
                elif include_third_parties:
                    party = Party.OTHER
                filename = "data/cache/" + name.replace(" ", "_") + ".txt"
                text = None
                if os.path.exists(filename):
                    with open(filename) as f:
                        text = ' '.join(list(f))
                corpus.append(WikipediaBiography(name, party, text))
                seen.add(name)
        return corpus

    def _preprocess(self, s: str) -> list[str]:
        s = re.sub(re.compile(r"\[\d*]"), "", s)  # remove footnotes
        result = []
        for token in word_tokenize(s):
            if token.isalpha():
                stem = self.stemmer.stem(token)
                if stem not in self.stop_words:
                    result.append(stem)
        return result

    def train(self, training_corpus: list[WikipediaBiography], custom_stop_words: int):
        # calculate priors
        v = len(training_corpus)
        parties = [doc.party for doc in training_corpus]
        for cd in self.data:
            cd.prior = parties.count(cd.party) / v

        # calculate raw ngram counts
        for doc in training_corpus:
            cd = self.data[doc.party.value]
            for ngram in self._preprocess(doc.text):
                if ngram not in cd.counts:
                    cd.counts[ngram] = 0
                cd.counts[ngram] += 1

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

        # generate custom stop words
        stop_words = {}
        for cd in self.data:
            for ngram in cd.counts:
                if ngram not in stop_words:
                    stop_words[ngram] = 0
                stop_words[ngram] += cd.counts[ngram]
        print("CUSTOM STOP WORDS:")
        top_ngrams = [t[0] for t in sorted(stop_words.items(), key=itemgetter(1), reverse=True)][:custom_stop_words]
        print("\t" + str(top_ngrams))
        self.stop_words.update(top_ngrams)

    def classify(self, dev_corpus: list[WikipediaBiography]):
        guesses = []
        for doc in dev_corpus:
            probabilities = []
            for cd in self.data:
                prob = np.log(cd.prior)
                for ngram in self._preprocess(doc.text):
                    if ngram in cd.likelihoods:
                        prob += np.log(cd.likelihoods[ngram])
                probabilities.append((prob, cd.party))
            guesses.append(max(probabilities)[1])
        return guesses

    @staticmethod
    def display_accuracy(guesses: list[Party], documents: list[WikipediaBiography]):
        correct = []
        incorrect = []
        for guess, doc in zip(guesses, documents):
            if guess == doc.party:
                correct.append(doc)
            else:
                incorrect.append(doc)
        print(f"Model made predictions for {len(documents)} documents.")
        print(f"CORRECT: {len(correct)}")
        print(f"INCORRECT: {len(incorrect)}")
        print(f"Model Accuracy: {len(correct) / len(documents)}")

    def exploration(self):
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
