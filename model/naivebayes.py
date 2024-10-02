import math
import re
import string

from nltk.stem import PorterStemmer, WordNetLemmatizer



class NaiveBayes(object):

    def __init__(self, prior_prob_offensive, prior_prob_non_offensive, bag_of_words, vocabulary_size,
                 total_words_in_offensive, total_words_in_nonoffensive):
        self.prior_prob_offensive = prior_prob_offensive
        self.prior_prob_non_offensive = prior_prob_non_offensive
        self.bag_of_words = bag_of_words
        self.vocabulary_size = vocabulary_size
        self.total_words_in_offensive = total_words_in_offensive
        self.total_words_in_nonoffensive = total_words_in_nonoffensive

    def predict(self, data, k=1):

        prob_without_prior_offensive = 1
        prob_without_prior_non_offensive = 1
        for i in range(len(data[0])):
            if data[0][i] in self.bag_of_words:

                word_occurrence_count_in_offensive_docs = self.bag_of_words[data[0][i]]['offensive_count']
                word_occurrence_count_in_non_offensive_docs = self.bag_of_words[data[0][i]]['non_offensive_count']

                prob_without_prior_offensive *= ((word_occurrence_count_in_offensive_docs + k) /
                                                 (self.total_words_in_offensive + self.vocabulary_size))

                prob_without_prior_non_offensive *= ((word_occurrence_count_in_non_offensive_docs + k) /
                                                     (self.total_words_in_nonoffensive + self.vocabulary_size))

            else:
                pass

        probability_with_prior_offensive = self.prior_prob_offensive * prob_without_prior_offensive
        probability_with_prior_non_offensive = self.prior_prob_non_offensive * prob_without_prior_non_offensive

        log_prob_offensive = math.log10(probability_with_prior_offensive)
        log_prob_non_offensive = math.log10(probability_with_prior_non_offensive)

        abs_log_prob_offensive = abs(log_prob_offensive)
        abs_log_prob_non_offensive = abs(log_prob_non_offensive)

        if log_prob_offensive > log_prob_non_offensive:
            return "offensive"
            # pass

        elif log_prob_offensive == log_prob_non_offensive:
            return "undecided"
            # pass
        else:
            return "nonoffensive"

    @classmethod
    def train(cls, data, k=1):

        total_docs = 0
        labels_dictionary = {}
        for i in range(len(data)):
            if data[i][1] in labels_dictionary:
                labels_dictionary[data[i][1]] = [data[i][1], labels_dictionary[data[i][1]][1]+1]
                total_docs += 1
                pass
            else:
                labels_dictionary[data[i][1]] = [data[i][1], 1]
                total_docs += 1
                pass

        total_offensive_docs = labels_dictionary['offensive'][1]
        total_non_offensive_docs = labels_dictionary['nonoffensive'][1]

        print('\nTotal Offensive Documents: ', total_offensive_docs)
        print('Total Non-offensive Documents: ', total_non_offensive_docs)
        print('Total Documents: ', total_docs)

        print("\nFinal arbitrary number of classes: ", labels_dictionary)

        bag_of_words = {}
        total_words_in_offensive = 0
        total_words_in_nonoffensive = 0
        total_words = 0

        for i in range(len(data)):
            for j in range(len(data[i][0])):
                if data[i][0][j] in bag_of_words:

                    if data[i][1] == 'offensive':
                        bag_of_words[data[i][0][j]]['count'] = bag_of_words[data[i][0][j]]['count'] + 1
                        bag_of_words[data[i][0][j]]['offensive_count'] = bag_of_words[data[i][0][j]]['offensive_count'] + 1
                        total_words_in_offensive += 1
                        total_words += 1
                    else:
                        bag_of_words[data[i][0][j]]['count'] = bag_of_words[data[i][0][j]]['count'] + 1
                        bag_of_words[data[i][0][j]]['non_offensive_count'] = bag_of_words[data[i][0][j]]['non_offensive_count'] + 1
                        total_words_in_nonoffensive += 1
                        total_words += 1
                else:
                    if data[i][1] == 'offensive':
                        bag_of_words[data[i][0][j]] = {'count': 1, 'non_offensive_count': 0, 'offensive_count': 1}
                        total_words_in_offensive += 1
                        total_words += 1
                    else:
                        bag_of_words[data[i][0][j]] = {'count': 1, 'non_offensive_count': 1, 'offensive_count': 0}
                        total_words_in_nonoffensive += 1
                        total_words += 1
                pass

        vocabulary_size = len(bag_of_words)
        print("\nTotal Vocabulary size: ", vocabulary_size)
        print("Total Words: ", total_words)
        print("Total Words in Offensive Docs: ", total_words_in_offensive)
        print("Total Words in Non-offensive Docs: ", total_words_in_nonoffensive)

        count_of_offensive_docs = labels_dictionary['offensive'][1]
        count_of_non_offensive_docs = labels_dictionary['nonoffensive'][1]

        count_of_total_docs = count_of_offensive_docs+count_of_non_offensive_docs

        prior_prob_offensive_docs = (count_of_offensive_docs / count_of_total_docs)
        prior_prob_non_offensive_docs = (count_of_non_offensive_docs / count_of_total_docs)

        print("\nPrior Probability for Offensive Docs: ", prior_prob_offensive_docs)
        print("Prior Probability for Non-offensive Docs: ", prior_prob_non_offensive_docs)

        correct_predictions = 0
        incorrect_predictions = 0
        for i in range(len(data)):
            prob_without_prior_offensive = 1
            prob_without_prior_non_offensive = 1
            for j in range(len(data[i][0])):
                if data[i][0][j] in bag_of_words:

                    word_occurrence_count_in_offensive_docs = bag_of_words[data[i][0][j]]['offensive_count']
                    word_occurrence_count_in_non_offensive_docs = bag_of_words[data[i][0][j]]['non_offensive_count']

                    prob_without_prior_offensive *= ((word_occurrence_count_in_offensive_docs + k) /
                                                     (total_words_in_offensive + vocabulary_size))
                    
                    prob_without_prior_non_offensive *= ((word_occurrence_count_in_non_offensive_docs + k) /
                                                         (total_words_in_nonoffensive + vocabulary_size))

                else:
                    pass

            probability_with_prior_offensive = prior_prob_offensive_docs * prob_without_prior_offensive
            probability_with_prior_non_offensive = prior_prob_non_offensive_docs * prob_without_prior_non_offensive
            
            log_prob_offensive = math.log10(probability_with_prior_offensive)
            log_prob_non_offensive = math.log10(probability_with_prior_non_offensive)

            abs_log_prob_offensive = abs(log_prob_offensive)
            abs_log_prob_non_offensive = abs(log_prob_non_offensive)

            if log_prob_offensive > log_prob_non_offensive:
                if data[i][1] == 'offensive':
                    correct_predictions += 1
                else:
                    incorrect_predictions += 1
                    pass

            elif log_prob_offensive == log_prob_non_offensive:
                pass
            else:
                if data[i][1] == 'nonoffensive':
                    correct_predictions += 1
                    pass
                else:
                    incorrect_predictions += 1
                    pass

        print('\nCorrect predictions: ', correct_predictions)
        print('Incorrect predictions: ', incorrect_predictions)

        return cls(
            prior_prob_offensive_docs,
            prior_prob_non_offensive_docs,
            bag_of_words,
            vocabulary_size,
            total_words_in_offensive,
            total_words_in_nonoffensive
        )


def features1(data, k=1):

    total_docs = 0
    labels_dictionary = {}
    for i in range(len(data)):
        if data[i][1] in labels_dictionary:
            labels_dictionary[data[i][1]] = [data[i][1], labels_dictionary[data[i][1]][1] + 1]
            total_docs += 1
            pass
        else:
            labels_dictionary[data[i][1]] = [data[i][1], 1]
            total_docs += 1
            pass

    total_offensive_docs = labels_dictionary['offensive'][1]
    total_non_offensive_docs = labels_dictionary['nonoffensive'][1]

    print('\nTotal Offensive Documents: ', total_offensive_docs)
    print('Total Non-offensive Documents: ', total_non_offensive_docs)
    print('Total Documents: ', total_docs)
    print("\nLabel Dictionary: ", labels_dictionary)

    stop_words_regex = r'\b(?:the|and|is|in|of|it|to|with|for|as|but|this|at|on|by|an|or|not|be|are)\b'
    bag_of_words = {}
    total_words_in_offensive = 0
    total_words_in_nonoffensive = 0
    total_words = 0

    for i in range(len(data)):
        for j in range(len(data[i][0])):
            if re.match(stop_words_regex, data[i][0][j], flags=re.IGNORECASE):
                pass

            else:
                if data[i][0][j] in bag_of_words:
                    if data[i][1] == 'offensive':
                        bag_of_words[data[i][0][j]]['count'] = bag_of_words[data[i][0][j]]['count'] + 1
                        bag_of_words[data[i][0][j]]['offensive_count'] = bag_of_words[data[i][0][j]][
                                                                             'offensive_count'] + 1
                        total_words_in_offensive += 1
                        total_words += 1
                        pass

                    else:
                        bag_of_words[data[i][0][j]]['count'] = bag_of_words[data[i][0][j]]['count'] + 1
                        bag_of_words[data[i][0][j]]['non_offensive_count'] = bag_of_words[data[i][0][j]][
                                                                                 'non_offensive_count'] + 1
                        total_words_in_nonoffensive += 1
                        total_words += 1
                        pass

                else:
                    if data[i][1] == 'offensive':
                        bag_of_words[data[i][0][j]] = {'count': 1, 'non_offensive_count': 0, 'offensive_count': 1}
                        total_words_in_offensive += 1
                        total_words += 1
                    else:
                        bag_of_words[data[i][0][j]] = {'count': 1, 'non_offensive_count': 1, 'offensive_count': 0}
                        total_words_in_nonoffensive += 1
                        total_words += 1
                    pass
                pass

    vocabulary_size = len(bag_of_words)
    count_of_offensive_docs = labels_dictionary['offensive'][1]
    count_of_non_offensive_docs = labels_dictionary['nonoffensive'][1]

    count_of_total_docs = count_of_offensive_docs + count_of_non_offensive_docs

    prior_prob_offensive_docs = (count_of_offensive_docs / count_of_total_docs)
    prior_prob_non_offensive_docs = (count_of_non_offensive_docs / count_of_total_docs)
    nb_instance = NaiveBayes(prior_prob_offensive_docs, prior_prob_non_offensive_docs, bag_of_words,
                             vocabulary_size, total_words_in_offensive, total_words_in_nonoffensive)
    return nb_instance


def features2(data, k=1):
    
    total_docs = 0
    labels_dictionary = {}
    for i in range(len(data)):
        if data[i][1] in labels_dictionary:
            labels_dictionary[data[i][1]] = [data[i][1], labels_dictionary[data[i][1]][1] + 1]
            total_docs += 1
            pass
        else:
            labels_dictionary[data[i][1]] = [data[i][1], 1]
            total_docs += 1
            pass

    total_offensive_docs = labels_dictionary['offensive'][1]
    total_non_offensive_docs = labels_dictionary['nonoffensive'][1]

    print('\nTotal Offensive Documents: ', total_offensive_docs)
    print('Total Non-offensive Documents: ', total_non_offensive_docs)
    print('Total Documents: ', total_docs)
    print("\nLabel Dictionary: ", labels_dictionary)

    stop_words_regex = r'\b(?:the|and|is|in|of|it|to|with|for|as|but|this|at|on|by|an|or|not|be|are)\b'
    bag_of_words = {}
    total_words_in_offensive = 0
    total_words_in_nonoffensive = 0
    total_words = 0
    lemma = WordNetLemmatizer()
    for i in range(len(data)):
        for j in range(len(data[i][0])):
            current_token = lemma.lemmatize(data[i][0][j])
            if re.match(stop_words_regex, data[i][0][j], flags=re.IGNORECASE):
                pass

            else:
                if lemma.lemmatize(data[i][0][j]) in bag_of_words:
                    if data[i][1] == 'offensive':
                        bag_of_words[current_token]['count'] = bag_of_words[current_token]['count'] + 1
                        bag_of_words[current_token]['offensive_count'] = bag_of_words[current_token][
                                                                             'offensive_count'] + 1
                        total_words_in_offensive += 1
                        total_words += 1
                        pass

                    else:
                        bag_of_words[current_token]['count'] = bag_of_words[current_token]['count'] + 1
                        bag_of_words[current_token]['non_offensive_count'] = bag_of_words[current_token][
                                                                                 'non_offensive_count'] + 1
                        total_words_in_nonoffensive += 1
                        total_words += 1
                        pass

                else:
                    if data[i][1] == 'offensive':
                        bag_of_words[current_token] = {'count': 1, 'non_offensive_count': 0, 'offensive_count': 1}
                        total_words_in_offensive += 1
                        total_words += 1
                    else:
                        bag_of_words[current_token] = {'count': 1, 'non_offensive_count': 1, 'offensive_count': 0}
                        total_words_in_nonoffensive += 1
                        total_words += 1
                    pass

    vocabulary_size = len(bag_of_words)
    count_of_offensive_docs = labels_dictionary['offensive'][1]
    count_of_non_offensive_docs = labels_dictionary['nonoffensive'][1]

    count_of_total_docs = count_of_offensive_docs + count_of_non_offensive_docs

    prior_prob_offensive_docs = (count_of_offensive_docs / count_of_total_docs)
    prior_prob_non_offensive_docs = (count_of_non_offensive_docs / count_of_total_docs)
    nb_instance = NaiveBayes(prior_prob_offensive_docs, prior_prob_non_offensive_docs, bag_of_words,
                             vocabulary_size, total_words_in_offensive, total_words_in_nonoffensive)
    return nb_instance
