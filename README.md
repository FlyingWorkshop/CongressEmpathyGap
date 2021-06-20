# CongressNLP
This project uses a Naive Bayes Classifier to predict whether wikipedia articles are about democrats or republicans or third party politicians.

There are 2 types of training (normal and special). The special training has the following results:

Unigram Accuracy: 0.7241379310344828
Bigram Accuracy: 0.7931034482758621

Special training generally performs better for both unigram and bigram predictions. The special training algorithm
is not based on the Naive Bayes algorithm and forgoes normalizing likelihoods and also excludes politicians with high
absolute value dw-nominates.


NOTE: This project was inspired by a class I took in 2021 with Prof. Dan Jurafsky; the original motivation, however, came from a paper I read in an English class
taught by Prof. Rebecca Richardson.
