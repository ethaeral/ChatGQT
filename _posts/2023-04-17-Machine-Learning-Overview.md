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
Steps:
1. Defining a problem
2. Creating a hypothesis
3. Create simple Heuristic
4. Measure Impact
5. More complex technique
6. Measure impact
7. Tune model
8. Decide which has better performance

### Naive Bayes
Based on Bayes' Theorem, it is known as a probabilistic classifier. Not great for predictions

Bayes theorem describes the probability of an event, based on prior knowledge of conditions that might be related to the event. It is also known to give us the allowance to "invert" conditional probabilities.

It can be expressed mathematically as the following equation:
$$P(A|B)=\frac{P(B|A)P(A)}{P(B)}$$
where A and B are events and P(B) != 0

Given this example of having trying to predict whether an email is a spam or not here we can use the Naive Bayes Classifier, in this case is heuristic instead of optimal  
Probability chain rule

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
Accuracy is how many correct prediction vs false prediction
    - Only takes into account true positive and true negatives
    - Doesn't penalize false negatives or false negatives
Confusion Matrix is used can be used to differeniate between accuracy of two models

Sensitivity: True Positive / (True Positive + False Negative)
    - The models ability to classify correctly
    - Fewer false positives
Specificity:  True Negatives / True negatives + False Positives
    - The models ability to correctly classify legitimate cases
    - Fewer false negatives
Precision: True Positive / True Positive + False Positive
    - Percentage of true positives
F1 Score: 2*(sensitivity*precision) / (sensitivity+ precision)
    - Harmonic mean between sensitivity and precision

There is a trade off between specificity and sensitivity which can be balanced by adjusting the decision point or decision threshold

Labeled data is split into training set, validation set, and test set

Validation set gives an opportunity to tune the model

When we are tuning the decision point we can plot it a Receiver Operator Characteristic Curve

To find the optimal tuned decision point we need to find the equal distance between the two extreme

When comparing different models, just take the area under the curve, which ever value is higher, that model is the better predictor

Hyperparameter Tuning
Parameters are static variables that go with the model, like in laplace smoothing

Validation
    - Holdout Validation
    - Cross-Validation
        - K-Fold
            - Will find the most different ways to split the data into training set, validation set, and test set - train the model and compare the performance
        - Leave-one-out(k=n)
            - Best for small data, it is k fold but k-1 

### Naive Bayes Optimizations
When there are multiple variants to detect we can add another term into the formula

P(_|_) = ()/(P(K) * P([]K|K)) + (P(L) * P([]L|L)) + P(M) * P([]M|M)

Multiple priors and likelihood
*Likelihood will change

Naive Bayes Classifier has a few variants of its formula.
    - Gaussian
        - Supports continuous values 
        - Assumption that each class in normally distributed
            - Pros
                - Simple and fast
                - Solves multi-class problem -> good for identifying sentiment
                - Does well with few samples for training
                - Performs well in text analysis problems 
            - Cons
                - Relies on the assumption of independent features
                - Not ideal for large number of numerical attributes -> will cause high computation and will suffer the curse of dimensionality
                - If a category is not captured in the set, the model will assign 0 probability and you will be forced to use smoothing techniques
    - Multi-nomial
        - Event based model
        - Uses feature vectors represents frequencies
        - Detects the counts of events 
    - Bernoulli 
        - Event based model
        - Uses features that are independent boolean in binary form
        - Detects whether event is present or not
Techniques for NLP
    - TF-IDF (Term Frequency Inverse Document Frequency) score
        - Express importance and unimportance of a word

When the data shifts in categories we can retrain the model in specific intervals - we can incrementally train the model

*Feature Hashing
    - Do not need to retrain the model as much
    - Reduces targeted adjustments in vocabulary

### K-Nearest Neighbors
Goal: Understand the a label of an unknown based on their nearest 

Euclidean Distance
- Distance formula between each nodes

Picking K might be beneficial to pick an odd number

*Jaccard Distance
(#mismatches/#comparisons)
*Hamming Distance

Models based on distance will be sensitive to scaling

Improve model through adding weight and scaling the data
Computationally Expensive
    - Reduce computation -> KD Tree, binary tree that splits the N dimensional space into layers -> Suffer dimensionality growth pains
### Decision Trees
When the decision tree is being trained, given the data set it creates a tree where its split based on a threshold of a feature. 

Gini Impurity: Chance fo being incorrect if you randomly assign a label to an example in the same set 0-1

Information Gain = Starting Uncertainty - Average Uncertainty

Partitioning sets based on average uncertainty based on the feature classifier questions

To stop partitioning we can assign a max depth, assign minimum of numbers of examples in a node or go until the nodes are pure (although can cause over fitting the data)

How we would handle  
    - Missing data
        - Surrogate splits
    - Multi class labels
        - Same but adds the second label to impurity
    - Regression
        - Instead of gini impurity we're gonna used mean squared error ->  basically average MSE of all examples 

CART over fits easily
    - Limit max depth to 5
    - Boosting
        - Taking weak learners which are shallow depth trees and combine them into a stronger learner
        - Can overfit more, and needs cross validation
        - Using bagging
            - Bootstrap: You make multiple sample sets from the main data set and run it through the model (row sampling with replacement) -> Aggregation: then combines to one output
                - Trains the next model based off the first error
            - Random forest
                - Averages the output from the multiple models

C4.5 
    - Designed for classification
    - n-ary splits
    - information gain entropy


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
### Decision Point
Also known as a decision rule or threshold, is a cut off point in which anything below the cutoff is determined to be a certain class and anything above the cut off is the other class
### Accuracy
The number of true positives plus the number of the true negatives divided by the total number of examples
### Unbalanced Classes
When one class is far more frequently observed than another class
### Model Training
Determining the model parameter values
### Confusion Matrix
In the binary case a 2x2 matrix indicating hte number of true positive, true negatives, fake positives, and false negatives
### Sensitivity 
Also recall, is teh proportion of true positive which are correctly classified
### Specificity 
The proportion of true negatives which are correctly classified
### Precision
The number of true positives divided by the true positives plus false positives
### F1 Precision 
The harmonic mean of the precision and recall
### Validation
The technique of holding out some portion of examples to be tested separately from the set of examples used to train the model
### Generalize
The ability of a model to perform well on the test set as well as examples beyond the test set
### Receiver Operator Characteristic Curve
Also ROC curve, is a plot of how the specificity and sensitivity change as the decision threshold changes. The area under the ROC curve, or AUC, is the probability 
### Hyperparameter
Any parameter associated with a mode which is not learned
### Multinomial Distribution
A distribution which models the probability of counts of particular outcomes
### TF-IDF
Short for Term Frequency-Inverse Document Frequency, TF-IDF is a method of transforming features, usually representing counts of words, into values representing the overall importance of different words across some documents
### Online Learning
Incremental learning within a model to represent an incrementally changing population
### N-Gram
A series of adjacent words of length n
### Feature Hashing
Representing feature inputs such as articles or messages, as the results of hashes modded by predetermined value
### scikit-learn
A machine learning python library which provides implementations of regression, classification, and clustering
### Kernel Density Estimation
Also KDE, a way to estimate the probability distribution of some data.
### Cluster
A consolidated group of points
### Euclidean Distance
The length of the line between two points
### Feature Normalization
Typically referring to feature scaling that places the values of a feature between 0 and 1
### Feature Standardization
Typically referring to feature scaling that centers the values of a feature around the mean of the feature and scales by the standard deviation of the feature
### Jaccard Distance
One minus the ratio of the number of like binary feature values and the number of like and unlike binary feature values, excluding instances of matching zeros
### Simple Matching Distance
One minus the ratio of the number of like binary feature values and the number of like and unlike binary feature values
### Manhattan Distance
Sum of the absolute difference of two input features
### Hamming Distance
The sum of the non-matching categorical feature values
### Decision Tree
A tree-based model which traverses examples down to a leaf nodes by the properties of the examples features.
### Sample Size
The number of observation taken from a complete population
### Classification and Regression Tree
Also CART, is an algorithm for constructing an approximate optimal decision tree for given examples
### Missing Data 
When some features within an example are missing
### Split Point 
A pair of feature and feature value