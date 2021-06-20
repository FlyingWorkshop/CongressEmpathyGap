from nltk import bigrams
from utils.document import Document
from utils.party import Party
from typing import Type
import math


class Corpus:
    def __init__(self, label: Party):
        self.label = label
        self.num_docs = 0
        self.prior = 0

        self.unigram_counts = {}
        self.unigram_likelihoods = {}
        self.unigram_size = 0

        self.bigram_counts = {}
        self.bigram_likelihoods = {}
        self.bigram_size = 0

    def process(self, documents: list[Type[Document]], unigram_vocab: set, bigram_vocab: set):
        for doc in documents:
            if doc.label != self.label:
                continue
            self.num_docs += 1
            ngram_count(doc.tokens, self.unigram_counts, unigram_vocab)
            ngram_count(bigrams(doc.tokens), self.bigram_counts, bigram_vocab)
        self.unigram_size = len(self.unigram_counts)
        self.bigram_size = len(self.bigram_counts)
        self.prior = self.num_docs / len(documents)

    def train_likelihoods(self, corpora_unigram_vocab: set, corpora_bigram_vocab: set):
        self._train_unigram_likelihoods(corpora_unigram_vocab)
        self._train_bigram_likelihoods(corpora_bigram_vocab)

    def _train_unigram_likelihoods(self, corpora_unigram_vocab: set):
        denominator = len(corpora_unigram_vocab) + len(self.unigram_counts)
        for unigram, count in self.unigram_counts.items():
            self.unigram_likelihoods[unigram] = (1 + count) / denominator

    def _train_bigram_likelihoods(self, corpora_bigram_vocab: set):
        v = len(corpora_bigram_vocab)
        for bigram, bigram_count in self.bigram_counts.items():
            unigram_count = self.unigram_counts[bigram[1]]
            self.bigram_likelihoods[bigram] = (1 + bigram_count) / (unigram_count + v)

    def ngram_score(self, ngrams, likelihoods: dict):
        if self.prior == 0:
            return -math.inf
        score = math.log(self.prior)
        for ngram in ngrams:
            if ngram in likelihoods:
                score += math.log(likelihoods[ngram])
        return score

    def display(self):
        print(f"Corpus Label: {self.label}")
        print(f"Num Docs: {self.num_docs}")
        print(f"Prior: {self.prior}")
        print(f"Unigram Vocab Size: {self.unigram_size}")
        print(f"Bigram Vocab Size: {self.bigram_size}")
        # print(f"Top Unigrams: {self.label}")
        # print(f"Top Bigrams: {self.label}")


def ngram_count(ngrams, count: dict, vocab: set, def_val=0, inc_val=1):
    for ngram in ngrams:
        if ngram not in count:
            vocab.add(ngram)
            count[ngram] = def_val
        count[ngram] += inc_val