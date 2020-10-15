#NLP Language Modeling

##Instructions
1. Verify ``main.py`` is in your current directory.
2. Launch terminal.
3. ``cd`` to proper directory where ``main.py`` is located.
4. Run ``python main.py`` in your terminal.
5. Open ``output.txt`` to obtain results to questions ``Part I`` and ``Part II``. 
6. Open ``test_PP.txt`` and ``train_PP.txt`` to obtain pre-processed results after Step 1.1.
7. Open ``train_unigram_lm.txt`` and  ``train_bigram_lm.txt`` to obtain language model results after Step 1.2.

This project trains the following language models and evaluates them on a test corpora with an additional separate sentence.
1. A unigram maximum likelihood model.
2. A bigram maximum likelihood model.
3. A bigram model with Add-One smoothing.

## Test  Corpora
1. train.txt
2. test.txt

## 1.1 Pre-processing
Prior to training, complete the following pre-processing steps:
1. Pad each sentence in the training and test corpora with start and end symbols (you can
use <s> and </s>, respectively).
2. Lowercase all words in the training and test corpora. Note that the data already has
been tokenized (i.e. the punctuation has been split off words).
3. Replace all words occurring in the training data once with the token <unk>. Every word
in the test data not seen in training should be treated as <unk>.

## 1.2 Training the models
Using train.txt, train the following language models:
1. A unigram maximum likelihood model.
2. A bigram maximum likelihood model.
3. A bigram model with Add-One smoothing.

## 1.3 Questions
1. How many word types (unique words) are there in the training corpus? Please include
the padding symbols and the unknown token.
2. How many word tokens are there in the training corpus?
3. What percentage of word tokens and word types in the test corpus did not occur in
training (before you mapped the unknown words to <unk> in training and test data)?
Please include the padding symbols in your calculations.
4. Now replace singletons in the training data with <unk> symbol and map words (in the
test corpus) not observed in training to <unk>. What percentage of bigrams (bigram
types and bigram tokens) in the test corpus did not occur in training (treat <unk> as a
regular token that has been observed).
5. Compute the log probability of the following sentence under the three models (ignore
capitalization and pad each sentence as described above). Please list all of the parameters
required to compute the probabilities and show the complete calculation. Which
of the parameters have zero values under each model? Use log base 2 in your calculations.
Map words not observed in the training corpus to the <unk> token.
• I look forward to hearing your reply .
6. Compute the perplexity of the sentence above under each of them odels.
7. Compute the perplexity of the entire test corpus under each of the models. Discuss the
differences in the results you obtained.

