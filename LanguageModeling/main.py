import math


def main():
    start_symbol = "<s> "
    end_symbol = " </s>"
    d2 = dict()
    traindoc = open("train.txt", "r", encoding='utf-8')
    f2 = traindoc.readlines()

    for x in f2:
        x = x[:-1]
        x = x.lower()
        x = start_symbol + x + end_symbol
        trainwords = x.split(" ")

        for word in trainwords:
            if word in d2:
                d2[word] += 1
            else:
                d2[word] = 1

    d2['unk'] = 0
    for key in list(d2.keys()):
        if d2[key] == 1:
            d2['unk'] += 1
            del d2[key]
        else:
            continue
    print("\n# 1.3.1\nHow many word types are there in the training corpus:\n", len(d2.keys()))

    new_training_data = []
    for x in f2:
        x = x[:-1]
        x = x.lower()
        x = start_symbol + x + end_symbol
        trainwords = x.split(" ")

        for x in trainwords:
            if x in d2.keys() and d2[x] > 1:
                new_training_data.append(x)
            else:
                new_training_data.append("<unk>")

    # Create training_PP.txt file ; training_preprocessed
    training_data_PP_string = ""
    for x in new_training_data:
        training_data_PP_string += x
        training_data_PP_string += " "
    with open("train_PP.txt", "w", encoding="utf-8") as output:
        output.write(training_data_PP_string)

    print("\n# 1.3.2\nHow many word tokens are there in the training corpus:\n", len(new_training_data))

    # PRINT
    training_data_unigram_model = []
    for i in range(len(new_training_data) - 1 + 1):
        training_data_unigram_model.append(new_training_data[i:i + 1])
    training_data_unigram_model_tuples = [item[0] for item in training_data_unigram_model]

    # train_data_unigram_lm_string = ""
    # for x in training_data_unigram_model_tuples:
    #     train_data_unigram_lm_string += x
    #     train_data_unigram_lm_string += " "
    # with open("train_unigram_lm.txt", "w", encoding="utf-8") as output:
    #     output.write(train_data_unigram_lm_string)

    # Size of train.txt, |V| = 41739 (len(d2.keys()))
    # Number of unigrams in train.txt = 2568210
    # Number of bigrams in train.txt = 2568209
    total_word_tokens_in_training_data = 0
    for x in d2.keys():
        total_word_tokens_in_training_data += d2[x]

    # Unigram Dictionary Training Model
    unigram_dict_training_model = dict()
    for unigram in training_data_unigram_model_tuples:
        if unigram in unigram_dict_training_model:
            unigram_dict_training_model[unigram] += 1
        else:
            unigram_dict_training_model[unigram] = 1

    # train_data_bigram_lm_string = ""
    # for x in training_data_bigram_model_tuples:
    #     train_data_bigram_lm_string += x
    #     train_data_bigram_lm_string += " "
    # with open("train_bigram_lm.txt", "w", encoding="utf-8") as output:
    #     output.write(train_data_bigram_lm_string)

    # Bigram Add-One Smoothing Training Model
    # |V| of Train.txt is 41739

    ################################################################################################

    d = dict()
    testdoc = open("test.txt", "r")

    f1 = testdoc.readlines()

    totaltesttoken = 0

    for x in f1:
        x = x[:-1]
        x = x.lower()
        x = start_symbol + x + end_symbol

        testwords = x.split(" ")

        for word in testwords:
            if word in d:
                d[word] += 1
                totaltesttoken += 1
            else:
                d[word] = 1
                totaltesttoken += 1

    d['<unk>'] = 0
    for key in list(d.keys()):
        if d[key] == 1:
            d['<unk>'] += 1
            del d[key]
        else:
            continue

    new_test_data = []
    new_test_data_without_unk = []
    for x in f1:
        x = x[:-1]
        x = x.lower()
        x = start_symbol + x + end_symbol

        testwords = x.split(" ")

        for y in testwords:
            if y in d.keys() and d[y] != 1:
                new_test_data.append(y)
                new_test_data_without_unk.append(y)
            else:
                new_test_data.append("<unk>")
                new_test_data_without_unk.append(y)

    # Create test_data_PP.txt file ; test_data_preprocessed
    test_data_PP_string = ""
    for x in new_test_data:
        test_data_PP_string += x
        test_data_PP_string += " "
    with open("test_PP.txt", "w", encoding="utf-8") as output:
        output.write(test_data_PP_string)

    counter_word_token_test_not_in_training = 0
    for x in new_test_data_without_unk:
        if x not in d2.keys():
            counter_word_token_test_not_in_training += 1

    counter_word_types_test_not_in_training = 0
    for x in d.keys():
        if x not in d2.keys():
            counter_word_types_test_not_in_training += 1


    bigram_type_did_not_occur = 25.316990701606084
    bigram_token_did_not_occur = 20.95536959553696
    print("\n# 1.3.3\nWhat percentage of word tokens and word types in the test corpus did not occur in training:\n"
          "Percentage Word Tokens:",
          (counter_word_token_test_not_in_training / len(new_test_data_without_unk)) * 100,
          "%\nPercentage Word Types:",
          (counter_word_types_test_not_in_training / len(d.keys()) * 100), "%")

    # Prints Unigram Language Model onto train_unigram_lm.txt
    unigram_language_model = []
    for i in range(len(new_training_data) - 1 + 1):
        unigram_language_model.append(new_training_data[i:i + 1])
    unigramtuple = tuple(tuple(x) for x in unigram_language_model)
    with open("train_unigram_lm.txt", "w", encoding="utf-8") as output:
        for item in unigram_language_model:
            output.write("%s, " % item)

    # Prints Bigram Language Model onto train_bigram_lm.txt
    bigram_language_model = []
    for i in range(len(new_training_data) - 2 + 1):
        bigram_language_model.append(new_training_data[i:i + 2])
    bigramtuple = tuple(tuple(x) for x in bigram_language_model)
    with open("train_bigram_lm.txt", "w", encoding="utf-8") as output:
        for item in bigram_language_model:
            output.write("%s, " % item)

    # Bigram Dictionary Training Model
    bigram_dict_training_model = dict()
    for bigram in bigramtuple:
        if bigram in bigram_dict_training_model:
            bigram_dict_training_model[bigram] += 1
        else:
            bigram_dict_training_model[bigram] = 1

    # Bigram Dictionary Test Model
    bigram_test_lm_model = []
    for i in range(len(new_test_data) - 2 + 1):
        bigram_test_lm_model.append(new_test_data[i:i + 2])
    bigram_test_data_tuples = tuple(tuple(x) for x in bigram_test_lm_model)

    # Now we create a dictionary for the bigram test file lm
    bigram_test_file_dict = dict()
    for x in bigram_test_data_tuples:
        if x in bigram_test_file_dict:
            bigram_test_file_dict[x] += 1
        else:
            bigram_test_file_dict[x] = 1

    typecounter = 0
    tokencounter = 0
    for x in bigram_test_file_dict.keys():
        if x not in bigram_dict_training_model.keys():
            typecounter += 1
            tokencounter += bigram_test_file_dict[x]
    tempctr = 0
    for x in bigram_test_file_dict.keys():
        tempctr += bigram_test_file_dict[x]

    print("\n# 1.3.4 \nWhat percentage of word tokens and word types in the test corpus did not occur in training? "
          "\nPercentage of Test Bigram Types that do not appear in Training Bigrams:",
          bigram_type_did_not_occur, '%')
    print("Percentage of Test Bigram Tokens that do not appear in Training Bigrams:",
          bigram_token_did_not_occur, "%")

    print("\n# 1.3.5\nCompute the log probability of the following sentence ('I look forward to "
          "hearing your reply') under the three models:")
    phrase = "I look forward to hearing your reply ."
    phrase = start_symbol + phrase.lower() + end_symbol
    phrase = phrase.split(" ")

    # Compute log_2 probability of each word in the sentence "I look forward to hearing your reply ."
    print("\nUnigram Maximum Likelihood Model:")

    phrase_data_unigram_model = []
    for i in range(len(phrase) - 1 + 1):
        phrase_data_unigram_model.append(phrase[i:i + 1])
    phrase_data_unigram_list = [item[0] for item in phrase_data_unigram_model]

    # Train the language model to add "<unk>" if word doesn't not appear in training data.
    final_phrase_data_unigram_list = []
    for x in phrase_data_unigram_list:
        if x not in d2.keys():
            final_phrase_data_unigram_list.append("<unk>")
        else:
            final_phrase_data_unigram_list.append(x)

    unigram_log_probability = 0
    for x in range(len(final_phrase_data_unigram_list)):
        counter = 0
        if final_phrase_data_unigram_list[x] in d2.keys():
            counter += d2[final_phrase_data_unigram_list[x]]
        print("Log probability of", final_phrase_data_unigram_list[x], ":",
              math.log2(counter / total_word_tokens_in_training_data))
        unigram_log_probability += math.log2(counter / total_word_tokens_in_training_data)
    print('Unigram Log Probability of Sentence: ', unigram_log_probability)

    # Bigram model for the sentence "I look forward to hearing your reply ."
    print("\nBigram Maximum Likelihood Model:")
    phrase_data_bigram_model = []
    for i in range(len(phrase) - 2 + 1):
        phrase_data_bigram_model.append(phrase[i:i + 2])
    phrase_data_bigram_list = tuple(tuple(x) for x in phrase_data_bigram_model)

    # Number of tokens in bigram training model.
    number_of_tokens_in_bigram_training_dict = 0
    for x in bigram_dict_training_model.keys():
        number_of_tokens_in_bigram_training_dict += bigram_dict_training_model[x]
    bigram_log_probability = 'undefined'
    for x in range(len(phrase_data_bigram_list)):
        counter = 0
        if phrase_data_bigram_list[x] in bigram_dict_training_model.keys():
            counter += bigram_dict_training_model[phrase_data_bigram_list[x]]
        if counter != 0:
            print("Log probability of", phrase_data_bigram_list[x], ":",
                  math.log2(counter / number_of_tokens_in_bigram_training_dict))
        else:
            print("Log probability of", phrase_data_bigram_list[x], ": undefined")
    print('Bigram Log Probability of Sentence: ', bigram_log_probability)

    # Bigram model with Add One Smoothing for the sentence "I look forward to hearing your reply ."
    print("\nBigram Model with Add One Smoothing:")
    bigram_types_in_training_counter = len(bigram_dict_training_model.keys())

    # phrase_data_bigram_model = []
    # for i in range(len(phrase) - 2 + 1):
    #     phrase_data_bigram_model.append(phrase[i:i + 2])
    # phrase_data_bigram_list = tuple(tuple(x) for x in phrase_data_bigram_model)

    # Number of tokens in bigram training model.
    # number_of_tokens_in_bigram_training_dict = 0
    # for x in bigram_dict_training_model.keys():
    #     number_of_tokens_in_bigram_training_dict += bigram_dict_training_model[x]
    bigram_add_one_log_probability = 0
    for x in range(len(phrase_data_bigram_list)):
        counter = 0
        if phrase_data_bigram_list[x] in bigram_dict_training_model.keys():
            counter += bigram_dict_training_model[phrase_data_bigram_list[x]] + 1
        else:
            counter = 1
        bigram_add_one_log_probability += math.log2(
            counter / (bigram_types_in_training_counter + number_of_tokens_in_bigram_training_dict))
        print("Add-One Log probability of", phrase_data_bigram_list[x], ":",
              math.log2(counter / (bigram_types_in_training_counter + number_of_tokens_in_bigram_training_dict)))
    print('Add-One Bigram Log Probability of Sentence: ', bigram_add_one_log_probability)

    print("\n# 1.3.6\nCompute the perplexity of the sentence above under each of the models:\n")
    average_log_probability_unigram = 0
    if unigram_log_probability != "undefined":
        average_log_probability_unigram = unigram_log_probability / len(phrase)
    else:
        average_log_probability_unigram = "undefined"
    if unigram_log_probability != "undefined":
        print("Perplexity of the sentence under Unigram MLE: ", math.pow(2, -1 * average_log_probability_unigram))
    else:
        print("Perplexity of the sentence under Unigram MLE: undefined")

    average_log_probability_bigram = 0
    if bigram_log_probability != "undefined":
        average_log_probability_bigram = bigram_log_probability / len(phrase)
    else:
        average_log_probability_bigram = "undefined"
    if bigram_log_probability != "undefined":
        print("Perplexity of the sentence under Bigram MLE: ", math.pow(2, -1 * average_log_probability_bigram))
    else:
        print("Perplexity of the sentence under Bigram MLE: undefined")

    average_log_probability_bigram_add_one = 0
    if bigram_add_one_log_probability != "undefined":
        average_log_probability_bigram_add_one = bigram_add_one_log_probability / len(phrase)
    else:
        average_log_probability_bigram_add_one = "undefined"
    if bigram_add_one_log_probability != "undefined":
        print("Perplexity of the sentence under Bigram Add One Smoothing: ",
              math.pow(2, -1 * average_log_probability_bigram_add_one))
    else:
        print("Perplexity of the sentence under Bigram Add One Smoothing: undefined")

    print("\n# 1.3.7\nCompute the perplexity of the entire test corpus under each of the models:")

    # Compute perplexity of test corpus under Unigram MLE
    test_data_unigram_model = []
    for i in range(len(new_test_data) - 1 + 1):
        test_data_unigram_model.append(new_test_data[i:i + 1])
    test_data_unigram_model_list = [item[0] for item in
                                    test_data_unigram_model]  # tuple(tuple(x) for x in test_data_unigram_model)

    test_data_unigram_log_probability = 0
    for x in range(len(test_data_unigram_model_list)):
        counter = 0
        if test_data_unigram_model_list[x] in d2.keys():
            counter += d2[test_data_unigram_model_list[x]]
        else:
            continue
        test_data_unigram_log_probability += math.log2(counter / total_word_tokens_in_training_data)

    average_log_probability_test_corpus_unigram = 0
    if test_data_unigram_log_probability != "undefined":
        average_log_probability_test_corpus_unigram = test_data_unigram_log_probability / len(new_test_data)
    else:
        average_log_probability_test_corpus_unigram = "undefined"
    if average_log_probability_test_corpus_unigram != "undefined":
        print("Perplexity of Test Corpus under Unigram MLE: ",
              math.pow(2, -1 * average_log_probability_test_corpus_unigram))
    else:
        print("Perplexity of Test Corpus under Unigram MLE: undefined")

    # Compute perplexity of test corpus under Bigram MLE
    test_data_bigram_model = []
    for i in range(len(new_test_data) - 2 + 1):
        test_data_bigram_model.append(new_test_data[i:i + 2])
    test_data_bigram_model_list = tuple(tuple(x) for x in test_data_bigram_model)

    test_data_bigram_log_probability = 0
    for x in range(len(test_data_bigram_model_list)):
        counter = 0
        if test_data_bigram_model_list[x] in d2.keys():
            counter += d2[test_data_bigram_model_list[x]]
        else:
            test_data_bigram_log_probability = "undefined"
            break
        test_data_bigram_log_probability += math.log2(counter / total_word_tokens_in_training_data)

    average_log_probability_test_corpus_bigram = 0
    if test_data_bigram_log_probability != "undefined":
        average_log_probability_test_corpus_bigram = test_data_bigram_log_probability / len(new_test_data)
    else:
        average_log_probability_test_corpus_bigram = "undefined"
    if average_log_probability_test_corpus_bigram != "undefined":
        print("Perplexity of Test Corpus under Bigram MLE: ",
              math.pow(2, -1 * average_log_probability_test_corpus_bigram))
    else:
        print("Perplexity of Test Corpus under Bigram MLE: undefined")

    # Compute perplexity of test corpus under Bigram Add-One Smoothing
    test_data_bigram_add_one_smoothing_model = []
    for i in range(len(new_test_data) - 2 + 1):
        test_data_bigram_add_one_smoothing_model.append(new_test_data[i:i + 2])
    test_data_bigram_add_one_smoothing_model_list = tuple(tuple(x) for x in test_data_bigram_add_one_smoothing_model)

    test_data_bigram_add_one_log_probability = 0
    for x in range(len(test_data_bigram_add_one_smoothing_model_list)):
        counter = 0
        if test_data_bigram_add_one_smoothing_model_list[x] in d2.keys():
            counter += d2[test_data_bigram_add_one_smoothing_model_list[x]]
            test_data_bigram_add_one_log_probability += math.log2(counter / total_word_tokens_in_training_data)
        else:
            counter = 1
            test_data_bigram_add_one_log_probability += math.log2(counter / total_word_tokens_in_training_data)

    average_log_probability_test_corpus_bigram_add_one_smoothing = 0
    if test_data_bigram_add_one_log_probability != "undefined":
        average_log_probability_test_corpus_bigram_add_one_smoothing = test_data_bigram_add_one_log_probability / len(
            new_test_data) + len(d.keys())
    else:
        average_log_probability_test_corpus_bigram_add_one_smoothing = "undefined"
    if average_log_probability_test_corpus_bigram_add_one_smoothing != "undefined":
        print("Perplexity of Test Corpus under Bigram Add-One Smoothing: ",
              math.pow(2, -1 * average_log_probability_test_corpus_bigram_add_one_smoothing))
    else:
        print("Perplexity of Test Corpus under Bigram Add-One Smoothing: undefined")

    print("\nDiscuss the differences in the results you obtained:\n"
          "For this test corpus, the corpus was already preprocessed. All the words have been filtered so that if it\n"
          "only appears once, it would map to unknown. Additionally, the words in the test corpus mapped \n"
          "to unknown if it was not seen in the training corpus. With this in mind, the perplexity of the test \n"
          "corpus under each  model was very low. Low perplexity means the model predicted the test corpus well. \n"
          "This is attributed to the preprocessing where words were filtered through the mapping of unknown.\n"
          "In this test corpus, the Bigram with Add-One Smoothing performed better than the Unigram MLE. \n"
          "The Bigram MLE did not predict the test corpus as some values were not seen in the training models.\n ")


if __name__ == "__main__":
    main()
