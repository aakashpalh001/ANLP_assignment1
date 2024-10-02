# NLP Assignment

# Twitter Hate Speech Detection using Naive Bayes and Logistic Regression

## Project Overview

This project is focused on detecting hate speech in tweets using **Naive Bayes** and **Logistic Regression** classifiers. The dataset consists of tweets classified into "offensive" and "nonoffensive" categories. The objective is to build models that can accurately classify tweets based on their text content, and evaluate the performance of each model using accuracy and F1 scores.

## Dataset

The dataset used in this project contains preprocessed tweets with the following structure:

Example tweet:
```
([‘At’, ‘this’, ‘rate’, ‘,’, “I’m”, ‘going’, ‘to’, ‘be’, ‘making’, ‘slides’, ‘for’, ‘a’, ‘keynote’, ‘in’, ‘my’, ‘car’, ‘as’, ‘I’, ‘drive’, ‘home’, ‘.’], ‘nonoffensive’)
```
The data consists of:
- **Text**: Tokenized words in the tweet.
- **Label**: Either 'offensive' or 'nonoffensive'.

## Objectives

1. **Evaluation**: Implement functions to compute the **accuracy** and **F1 score** of a classifier.
2. **Naive Bayes Classifier**: 
   - Implement a Naive Bayes classifier from scratch with additive smoothing and log probabilities.
   - Evaluate its performance with varying smoothing parameters and visualize the results.
3. **Feature Engineering**:
   - Experiment with different feature sets, such as removing stop words, lemmatization, and bigrams, and test how they affect the Naive Bayes classifier.
4. **Logistic Regression Classifier**:
   - Implement a Logistic Regression classifier using gradient descent and L2 regularization.
   - Train the model and evaluate its performance on the dataset.

## Project Structure

```
.
├── data/
│   ├── NAACL_SRW_2016.csv
│   ├── NAACL_SRW_2016_tweets.json
├── model/
│   ├── init.py
│   ├── logreg.py            # Logistic Regression implementation
│   ├── naivebayes.py        # Naive Bayes implementation
├── assignment1.py           # Main script to run the models
├── evaluation.py            # Evaluation functions for accuracy and F1 score
├── helper.py                # Helper functions for feature engineering and training
├── utils.py                 # Utility functions for data processing
```

## Instructions

### 1. Naive Bayes Classifier
- Implement the Naive Bayes classifier in `model/naivebayes.py`.
- The classifier dynamically creates the vocabulary from training data and uses log probabilities to avoid underflow.
- Use the flag `--test_smooth` to run tests with different smoothing parameters.
- Example usage:
   ```bash
   python assignment1.py --naivebayes --test_smooth
   ```

### 2. Feature Engineering
- Implement two variants of feature engineering using the `features1()` and `features2()` functions.
- Test different techniques like stop word removal and n-grams.
- Use the flag `--feature_eng` to test these features:
   ```bash
   python assignment1.py --naivebayes --feature_eng
   ```

### 3. Logistic Regression Classifier
- Implement the Logistic Regression classifier in `model/logreg.py`.
- Use gradient descent with L2 regularization to optimize the model weights.
- Train the model and test it using the following command:
   ```bash
   python assignment1.py --logreg
   ```

## Evaluation

Both classifiers will be evaluated based on their accuracy and F1 score. The evaluation functions are implemented in `evaluation.py`. 

## Results and Discussion

- Compare the performance of the Naive Bayes and Logistic Regression models.
- Analyze how different smoothing parameters and feature engineering affect the Naive Bayes classifier.
- Discuss the results of logistic regression and its regularization.

## Requirements

To run this project, install the necessary libraries:
```bash
pip install numpy pandas matplotlib
```
