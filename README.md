# CongressNLP
This project uses a Naive Bayes Classifier to predict whether wikipedia articles are about democrats or republicans or third party politicians.

The current model is trained with a cache of Wikipedia biographies of various US governors. The accuracy for the unigram model with stop word filtering
is around 52%. Running the program from main also allows users to explore the data (e.g. if you're curious as to how many times the word "green" or "tax" appears
in varsious wikipedia articles; the exploration feature hopefully provides greater transparency into how the model actually works.
