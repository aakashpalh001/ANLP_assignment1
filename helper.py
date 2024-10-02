import matplotlib.pyplot as plt

from model.naivebayes import NaiveBayes, features1, features2
from model.logreg import LogReg, featurize
from evaluation import accuracy, f_1


def train_smooth(train_data, test_data):
  
    plt.xlabel("K")
    plt.ylabel("Accuracy")

    k_values = list(range(1, 30))
    accuracy_values = []

    for k in k_values:
        nb = NaiveBayes.train(train_data, k)
        model_acc = accuracy(nb, test_data, k)
        accuracy_values.append(model_acc)

    plt.plot(k_values, accuracy_values)
    plt.show()

    print("Smoothed Naive Bayes Model Accuracy for Different K Values:")
    for k, acc in zip(k_values, accuracy_values):
        print(f"K = {k}: Accuracy = {acc:.4f}")
    # print("Accuracy vals: ", accuracy_values)
    # print("K vals: ", k_values)
    plt.ylabel("f1_score")

    k_values = list(range(1, 30))
    accuracy_values = []

    for k in k_values:
        nb = NaiveBayes.train(train_data, k)
        f1_accuracy = f_1(nb, test_data, k)
        accuracy_values.append(f1_accuracy)

    plt.plot(k_values, accuracy_values)
    plt.show()

    print("Smoothed Naive Bayes F1 Score for Different K Values:")
    for k, acc in zip(k_values, accuracy_values):
        print(f"K = {k}: F1 Score = {acc:.4f}")


def train_feature_eng(train_data, test_data):

    classifier1 = features1(train_data)
    total_count = len(test_data)
    model_accuracy = 0
    for i in range(len(test_data)):
        predict = classifier1.predict(test_data[i])
        if test_data[i][1] == predict:
            model_accuracy += 1
            pass

    print("\nMODEL ACCURACY WITH FEATURE ENGINEERING 1:")
    print(f"Accuracy = {model_accuracy / total_count:.4f}")

    classifier2 = features2(train_data)
    total_count = len(test_data)
    model_accuracy = 0
    for i in range(len(test_data)):
        predict = classifier2.predict(test_data[i])
        if test_data[i][1] == predict:
            model_accuracy += 1
            pass

    print("\nMODEL ACCURACY WITH FEATURE ENGINEERING 2:")
    print(f"Accuracy = {model_accuracy / total_count:.4f}")



def train_logreg(train_data, test_data):
    # YOUR CODE HERE
    #     TODO:
    #         1) First, assign each word in the training set a unique integer index
    #         with `buildw2i()` function (in model/logreg.py, not here)
    #         2) Now that we have `buildw2i`, we want to convert the data into
    #         matrix where the element of the matrix is 1 if the corresponding
    #         word appears in a document, 0 otherwise with `featurize()` function.
    #         3) Train Logistic Regression model with the feature matrix for 10
    #         iterations with default learning rate eta and L2 regularization
    #         with parameter C=0.1.
    #         4) Evaluate the model on the test set.
    ########################### STUDENT SOLUTION ########################
    pass
    #####################################################################
