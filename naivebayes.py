from utils.corpus import Corpus, Party, ngram_count
from nltk import bigrams
from utils.util import index_max, get_label, top_n


class NaiveBayesClassifier:
    def __init__(self, use_third_parties=False):
        self.data = [Corpus(label) for label in Party]
        if not use_third_parties:
            self.data.pop()

    def get_corpus(self, label: Party):
        return self.data[label.value]

    def train(self, documents):
        unigram_vocab, bigram_vocab = set(), set()
        for corpus in self.data:
            corpus.process(documents, unigram_vocab, bigram_vocab)
        for corpus in self.data:  # wait for corpora vocabs to fully populate before calculating likelihoods
            corpus.train_likelihoods(unigram_vocab, bigram_vocab)

    def special_train(self, documents):
        unigram_vocab, bigram_vocab = set(), set()
        for corpus in self.data:
            for doc in documents:
                if doc.label != corpus.label or not doc.loyalty or not doc.dw:
                    continue
                corpus.num_docs += 1
                if abs(doc.dw) > 0.75:
                    continue
                inc = doc.loyalty
                ngram_count(doc.tokens, corpus.unigram_counts, unigram_vocab, 0, 1)
                ngram_count(bigrams(doc.tokens), corpus.bigram_counts, bigram_vocab, 0, inc)

        unigram_v = len(unigram_vocab)
        for corpus in self.data:
            corpus.unigram_size = len(corpus.unigram_counts)
            corpus.bigram_size = len(corpus.bigram_counts)
            corpus.prior = corpus.num_docs / len(documents)
            denominator = len(corpus.unigram_counts) + unigram_v
            for unigram, count in corpus.unigram_counts.items():
                corpus.unigram_likelihoods[unigram] = count
            for bigram, count in corpus.bigram_counts.items():
                corpus.bigram_likelihoods[bigram] = count


    def score(self, documents):
        scores = []
        for doc in documents:
            doc_scores = []
            for corpus in self.data:
                unigram_score = corpus.ngram_score(doc.tokens, corpus.unigram_likelihoods)
                bigram_score = corpus.ngram_score(bigrams(doc.tokens), corpus.bigram_likelihoods)
                doc_scores.append((unigram_score, bigram_score))
            scores.append(doc_scores)
        return scores

    def classify(self, scores: list[list[tuple]]):
        predictions = []
        for doc_scores in scores:
            unigram_scores = [t[0] for t in doc_scores]
            bigram_scores = [t[1] for t in doc_scores]
            unigram_pred = get_label(index_max(unigram_scores))
            bigram_pred = get_label(index_max(bigram_scores))
            predictions.append((unigram_pred, bigram_pred))
        return predictions

    def display(self, scores, predictions, documents):
        total_preds = 0
        rep_actual = 0
        uni_rep_pred = 0
        bi_rep_pred = 0
        dem_actual = 0
        uni_dem_pred = 0
        bi_dem_pred = 0
        uni_true_dem = 0
        uni_false_dem = 0
        uni_true_rep = 0
        uni_false_rep = 0
        bi_true_dem = 0
        bi_false_dem = 0
        bi_true_rep = 0
        bi_false_rep = 0
        for doc_score, doc_prediction, doc in zip(scores, predictions, documents):
            if not doc.tokens:
                continue
            total_preds += 1
            if doc.label == Party.REPUBLICAN:
                rep_actual += 1
            elif doc.label == Party.DEMOCRAT:
                dem_actual += 1

            correct_unigram_pred = doc_prediction[0] == doc.label
            correct_bigram_pred = doc_prediction[1] == doc.label
            if doc_prediction[0] == Party.DEMOCRAT:
                uni_dem_pred += 1
                if correct_unigram_pred:
                    uni_true_dem += 1
                else:
                    uni_false_dem += 1
            elif doc_prediction[0] == Party.REPUBLICAN:
                uni_rep_pred += 1
                if correct_unigram_pred:
                    uni_true_rep += 1
                else:
                    uni_false_rep += 1
            if doc_prediction[1] == Party.DEMOCRAT:
                bi_dem_pred += 1
                if correct_bigram_pred:
                    bi_true_dem += 1
                else:
                    bi_false_dem += 1
            elif doc_prediction[1] == Party.REPUBLICAN:
                bi_rep_pred += 1
                if correct_bigram_pred:
                    bi_true_rep += 1
                else:
                    bi_false_rep += 1

            doc.display()
            unigram_scores = [t[0] for t in doc_score]
            bigram_scores = [t[1] for t in doc_score]
            print(f"Unigram Scores: {unigram_scores}")
            print(f"Bigram  Scores: {bigram_scores}")
            print(f"Unigram Prediction: {doc_prediction[0].name}")
            print(f"Bigram  Prediction: {doc_prediction[1].name}")
            print()
        print("----")
        for corpus in self.data:
            corpus.display()
        print("----")
        print(f"Made {total_preds} predictions for {len(documents)} documents.")
        print(f"# Actual Republicans: {rep_actual}")
        print(f"# Actual Democrats: {dem_actual}")
        print("Unigrams: ")
        print(f"\t Correctly predicted {uni_true_rep} out of {rep_actual} actual Republicans.")
        print(f"\t Correctly predicted {uni_true_dem} out of {dem_actual} actual Democrats.")
        print(f"\t Predicted {uni_rep_pred} Republicans with {uni_true_rep} correct and {uni_false_rep} incorrect")
        print(f"\t Predicted {uni_dem_pred} Democrats with {uni_true_dem} correct and {uni_false_dem} incorrect.")
        print(f"\t Republican Recall: {uni_true_rep / rep_actual}")
        print(f"\t Republican Precision: {uni_true_rep / uni_rep_pred}")
        print(f"\t Democrat Recall: {uni_true_dem / dem_actual}")
        print(f"\t Democrat Precision: {uni_true_dem / uni_dem_pred}")
        print("Bigrams: ")
        print(f"\t Correctly predicted {bi_true_rep} out of {rep_actual} actual Republicans.")
        print(f"\t Correctly predicted {bi_true_dem} out of {dem_actual} actual Democrats.")
        print(f"\t Predicted {bi_rep_pred} Republicans with {bi_true_rep} correct and {bi_false_rep} incorrect.")
        print(f"\t Predicted {bi_dem_pred} Democrats with {bi_true_dem} correct and {bi_false_dem} incorrect.")
        print(f"\t Republican Recall: {bi_true_rep / rep_actual}")
        print(f"\t Republican Precision: {bi_true_rep / bi_rep_pred}")
        print(f"\t Democrat Recall: {bi_true_dem / dem_actual}")
        print(f"\t Democrat Precision: {bi_true_dem / bi_dem_pred}")
        print("----")
        print(f"Unigram Accuracy: {(uni_true_rep + uni_true_dem)/total_preds}")
        print(f"Bigram Accuracy: {(bi_true_rep + bi_true_dem)/total_preds}")

