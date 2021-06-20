from utils.corpus import Corpus, Party
from nltk import bigrams
from utils.util import index_max


class NaiveBayesClassifier:
    def __init__(self):
        self.data = [Corpus(label) for label in Party]

    def get_corpus(self, label: Party):
        return self.data[label.value]

    def train(self, documents):
        unigram_vocab, bigram_vocab = set(), set()
        for corpus in self.data:
            corpus.process(documents, unigram_vocab, bigram_vocab)
        for corpus in self.data:  # wait for corpora vocabs to fully populate before calculating likelihoods
            corpus.train_likelihoods(unigram_vocab, bigram_vocab)

    def score(self, documents):
        scores = []
        for doc in documents:
            doc_score_list = []
            for corpus in self.data:
                unigram_score = corpus.ngram_score(doc.tokens, corpus.unigram_likelihoods)
                bigram_score = corpus.ngram_score(bigrams(doc.tokens), corpus.bigram_likelihoods)
                doc_score_list.append((unigram_score, bigram_score))
            scores.append(doc_score_list)
        return scores

    def classify(self, scores: list[list[tuple]]):
        unigram_predictions = []
        bigram_predictions = []
        for sublist in scores:
            unigram_scores = []
            bigram_scores = []
            for t in sublist:
                unigram_scores.append(t[0])
                bigram_scores.append(t[1])
            unigram_predictions.append(self.data[index_max(unigram_scores)].label)
            bigram_predictions.append(self.data[index_max(bigram_scores)].label)
        return unigram_predictions, bigram_predictions

    def display_accuracy(self, predictions, documents):
        accuracy = {corpus.label: {"tp": 0, "fp": 0} for corpus in self.data}
        for prediction, doc in zip(predictions, documents):
            if prediction == doc.label:
                accuracy[prediction]["tp"] += 1
            else:
                accuracy[prediction]["fp"] += 1
        for label in accuracy:
            print(label.name.upper())
            print(f"\t TP: {accuracy[label]['tp']}")
            print(f"\t FP: {accuracy[label]['fp']}")

