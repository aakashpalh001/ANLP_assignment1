import random


def accuracy(classifier, data, k=1):
    total_count = len(data)
    model_accuracy = 0

    for i in range(len(data)):
        predict = classifier.predict(data[i], k)
        if data[i][1] == predict:
            model_accuracy += 1
            pass
        # pass

    return model_accuracy / total_count


def f_1(classifier, data, k=1):

    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    for i in range(len(data)):
        predict = classifier.predict(data[i], k)
        if predict == data[i][1]:
            if data[i][1] == 'offensive':
                true_positive += 1
                pass
            else:
                true_negative += 1
                pass
            # pass
        else:
            if data[i][1] == 'offensive':
                false_positive += 1
                pass
            else:
                false_negative += 1
                pass
            # pass

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)

    f1_score = (2 * precision * recall) / (precision + recall)
    return f1_score
