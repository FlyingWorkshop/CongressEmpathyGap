from naivebayes import NaiveBayesClassifier
from utils.loader import get_loader

# directories
TRAINING_DATA = "data/sources/congress"
DEVELOPER_DATA = "data/sources/united_states_governors_1775_2020.csv"


def main():
    cong_loader = get_loader("congress")
    training_docs = cong_loader(TRAINING_DATA)
    gov_loader = get_loader("governor")
    dev_docs = gov_loader(DEVELOPER_DATA, timespan=(1990, 2100))

    classifier = NaiveBayesClassifier()
    classifier.train(training_docs)

    scores = classifier.score(dev_docs)
    unigram_predictions, bigram_predictions = classifier.classify(scores)

    classifier.display_accuracy(unigram_predictions, dev_docs)
    classifier.display_accuracy(bigram_predictions, dev_docs)


if __name__ == '__main__':
    main()