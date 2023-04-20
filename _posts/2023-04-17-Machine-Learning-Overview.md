---
title: Machine Learning Overview
date: 2023-04-17 00:00:00 -0500
categories: [Core]
tags: [foryou, interview prep]
---
----
##### 📑 **Requirements**:
###### Statistics
##### 😝 **Cool Level**:
###### 8/10
----
## Perquisites
Things to be familiar with: 
Machine Learning:
- Features
    - Observations
        - Continuous
        - Categorical
        - Ordinal
- Labels
    - Indicators of groupings
        - Continuous
        - Categorical
        - Ordinal
- Examples 
    - Features with a label

Math:
- Array / Vectors
- Matrix
    - Multiplication
    - Inverse
    - Transpose
- Polynomials
    - Line
    - Quadratic
- Derivatives
- Probability 
    - Conditional
    - Distribution
        - Gaussian
        - Uniform
        - Beta
## Supervised Learning

### Naive Bayes
Based on Bayes' Theorem, it is known as a probabilistic classifier.

Bayes theorem describes the probability of an event, based on prior knowledge of conditions that might be related to the event. It is also known to give us the allowance to "invert" conditional probabilities.

It can be expressed mathematically as the following equation:
$$P(A|B)=\frac{P(B|A)P(A)}{P(B)}$$
where A and B are events and P(B) != 0

Given this example of having trying to predict whether an email is a spam or not here we can use the Naive Bayes Classifier, in this case is heuristic instead of optimal  
Probability chain rul e

P(spam|list of words) > bayes theorem
P(spam|list of words) = (P(spam) * P(list of words|spam)) / ((P(spam) * P(list of words|spam)) + (P(not spam) * P(list of words|not spam))) 

Bottom part of the fraction is called a Evidence
The model we are using is called the Bernoulli Model
The whole fraction is called the posterior

Classifies if spam if these conditions are met and are true
(spam | list of words) = (spam) * (list of words | spam)

To figure out the probability if a word is in a spam message then we have to also account for all the other words in the model - that list would be called a vocabulary

We need to utilize the *probability chain rule. In this incase when using this we are assuming each word is independent to each other which helps with calculations it in turn increases the bias - which hinder the models ability to understand nuances in the data

This reduces down to multiplying of each word showing up in a spam message

If theres a probability where one word has the probability of 0 of showing up in spam in an email we can implement *Laplace smoothing which is basically adding 0.5 to word probability to every word

To pre process or featurize the data we would need to perform these steps before feeding it into the the model:

Tokenization: Removing white space, punctuation, then creating the a list of just tokens
Stop word removal: Remove word non important words
Non-alphabetic removal: Remove symbols that aren't words 
Stemming: Removes -ing, -es -> Studies -> Studi
Lemmatization: Removes -ing but with nuances -> Studies -> Study
    - Requires more computations

Some people use Lowercasing, but this can remove the marking or name nouns places

After feautrizing we can then vectorize the list, where we convert the value in to binary

Priors - P(Spam)
Likelihoods - P(list of words|spam)



### Performance
### Naive Bayes Optimizations
### K-Nearest Neighbors
### Decision Trees
### Linear Regression
### Logistic Regression
### Support Vector Machine
## Unsupervised Learning
### K-Means
### Singular Value Decomposition
## Deep Learning
### Neural Networks
### Convolutional Neural Networks
### Recurrent Neural Networks
### Generative Adversarial Neural Networks
## Recommendation Systems
### Collaborative and Content Based Filtering
## Ranking
### Learning to Rank

## Key Terms
### Ranking  
Machine Learning Model Based On: Attaching numeric values to candidates   
Goal: Order the candidates where the candidates are ordered from most likely to be interacted with to the least likely to be interacted with.   
### Supervised Learning  
Machine Learning Model Based On: Previously observed features and labels, structured data
Goal: Attach the most likely label to a group of features.  
### Unsupervised Learning  
Machine Learning Model Based On: Unlabeled examples 
Goal: To produce patterns from the data and discover something previously unknown about the unlabeled examples
### Deep Learning
Machine Learning Model Based On: A collection of connected units, used on unstructured data
Goal: Input an example which is parsed through hidden layers to perform unsupervised or supervised learning
### Recommendation Systems 
Machine Learning Model Based On: System of models
Goal: Present an item to user that user is likely to consume
### Features 
A column of data describing an observation. Can be continuous, categorical, or ordinal 
### Labels  
A name paired with a set of features, can be discrete or continuous - used in supervised models
### Examples  
Pair of features and labels  
### Dimensions  
The number of features with a particular example  
### [Vectors](https://en.wikipedia.org/wiki/Feature_(machine_learning))  
Feature vector is a list of features representing a particular example  
### [Matrix](https://en.wikipedia.org/wiki/Matrix_(mathematics))  
Array of values consisting of multiple rows and columns  
### [Matrix Transpose](https://www.mathsisfun.com/definitions/transpose-matrix-.html)  
An action that flips matrix over a diagonal
### [Polynomial](https://en.wikipedia.org/wiki/Polynomial) 
A function with more than one coefficient pair   
### [Derivative](https://en.wikipedia.org/wiki/Derivative)  
Indicates how much the output of a function will change with respect to a change in its input
### [Probability](https://www.mathsisfun.com/data/probability.html)  
How likely an event is to occur. This can be independent or conditional
### [Probability Distribution](https://en.wikipedia.org/wiki/Probability_distribution)  
A function that takes in an outcome and outputs the probability of that particular outcome occurring.
### [Gaussian Distribution](https://www.youtube.com/watch?v=RNmDyzYw7aQ)  
Also known as normal distribution, a very common type of probability distribution which fits many real world observations
### [Uniform Distribution](https://en.wikipedia.org/wiki/Continuous_uniform_distribution)   
A probability distribution in which each outcome is equally likely 
### Model 
An approximation of a relationship between an input and an output
### Heuristic
An approach to finding a solution which is typically faster but less accurate than some optimal solution
### Bernoulli Distribution
A distribution which evaluates a particular outcome as binary. In the Bernoulli Naive Bayes classifier case, a word was either in a message or not in a message, which is a binary representation 
### Prior 
Indicate the probability of particular class regardless of the feature of some example
### Likelihood
The probability of some features given a particular class
### Evidence
The denominator of the Naive Bayes classifiers
### Posterior
The probability of a class given some features
### Vocabulary
The list of words that the Naive Bayes classifier recognizes
### Laplace Smoothing
A type of additive smoothing which mitigates the chance of encountering zero probabilities within the Naive Bayes classifier
### Tokenization
The splitting of some raw textual input into individual words or elements
### Featurization
The process of transforming raw inputs into individual words or elements
### Vectorizer 
Used in a step of featurizing. It transform some input into something else. One example is binary vectorizer which transforms tokenized messages into a binary vector indicating which items in the vocabulary appear in the message
### Stop Word
A word typically discarded, which doesn't add much predictive value
### Stemming
Removing the ending modifiers of words, leaving the stem of the word
### Lemmatization 
A more calculated form of stemming which ensures proper lemma results from removing the word modifiers.