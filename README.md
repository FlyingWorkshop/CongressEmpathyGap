# CongressNLP
This project uses a Naive Bayes Classifier to predict whether wikipedia articles are about democrats or republicans or third party politicians.

The current model is trained with a cache of Wikipedia biographies of various US governors. The accuracy for the unigram model with stop word filtering
is around 52%. Running the program from main also allows users to explore the data (e.g. if you're curious as to how many times the word "green" or "tax" appears
in varsious wikipedia articles; the exploration feature hopefully provides greater transparency into how the model actually works.

NOTE: This project was inspired by a class I took in 2021 with Prof. Dan Jurafsky; the original motivation, however, came from a paper I read in an English class
taught by Prof. Rebecca Richardson.


TODO: implement bigrams, use twitter data, use sen/reps data and scale ngram count by dw-nominate
