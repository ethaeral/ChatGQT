---
title: Machine Learning Overview
date: 2023-04-17 00:00:00 -0500
categories: [Core]
tags: [forme, interview prep, notes]
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
Goal of Linear Regression is to create best fit line given the plotted values. With this line a user has a ability to plug in a value and receive a predicted value. To find the linear regression

y=ax+b
y - dependent variable / labels
x - independent variable / features
a - slope
b - y intercept

1. Finding A:
a = sum of (xi - avg x)*(yi -  avg y) / sum of (xi- avg x)^2
2. Checking P value of coefficient, if the p value is less than 0.05 then its seen as significant -> coef / standard error of a coefficient = t-statistic and then find that value in a t distribution
3. Check confidence interval of the coefficient:
coef - CI * SECoef - coef+CI * SECoef 
4. Interpretation - Correlation of feature and label, positive R is a positive correlation, negative R is a negative correlation, 0 correlation is no correlation
5. Check performance with R squared value, the explained difference of the variance, the higher Rsq then less unexplained variance
Rsq = 1-(var(errors)/var(y))
Adj R2= 1-(n-1/n-2)*(1-R^2)
6. When working with multivariate functions, collinearity needs to be considered - this is when there is a possibility where independent variables are have correlation - how we interpret the coefficient will have to change
7. To find collinear variables, we can look at VIF (Variance Inflation Factor) 
    1 - no collinearity
    1-5 - moderate
    >=5 - severe - need to use a mitigation strategy
Mitigation strategy example: centering
8. Feature interaction
If a independent variable has interaction than another we can describe that in the function by multiplying together as an interaction term - after the introduction of the interaction term, if the Rsq goes up and the p value turns significant
We can make an interaction term with an independent variable with itself

If you put too many terms you can over fit your data

Care for Simpson paradox


### Logistic Regression
Logistic regression is similar to linear regression, but instead of y being a predicted value it is a probability of an event occurring.

Sigmoid function or logistic regression 
p(y|x) = (1/(1+e^-(Bix+B0)))

1. Assign a loss
Loss(yhat, y) = ((y)log(yhat) + (1-y)log(1-yhat))difference in yhat and y
yhat is predictive value 
y is actual values
cross-entropy: Loss(yhat, y) = -[((y)log(yhat) + (1-y)log(1-yhat))]
2. Performance and optimization
Sum of cross-entropy loss / # elements
Average loss among coefficient relate to variables, and how can we use the loss as an indicator for optimization
When plotting Loss v the Logistic Regression, you can find points of optimization alone the line by calculating the slope of change. The goal is to get closer to minima, and when this happens the loss function has converged and loss has been minimized
You can find the derivative of the loss function by using
dloss/dB1 = sum of [yhati - yi]x1i 
This will find the minima of the curve
Gradient Descent:
When you combine the derivative of each coefficient and its loss in a vector it becomes a gradient which will tell us how to adjust each weight of each parameters of the weights
We adjust like this:
B0 = B0coeff - dB0eff and so on

Bi^th = Bi^t - r deltaBi

r being a learning rate, reduces the chance to overshoot the minima - choose an optimal learning rate so its gets quickly to the minima but doesn't miss it

Outcome of the predictive model, we can determine if something will happen given a threshold 

Follow up with cross validation to find the optimal threshold

TO find the decision boundary 
1 dim: -(b0/b1)
2 dim: -(b1x1-b0)

represent that in a model in the gradient**
vectorb = [d(loss)/d(b0)]

3. Interpretation of impact on coefficient on output
Odds ratio: e^B1
Percent changed odds: 1-e^b1

4. Multiple classes
When you want to predict between multiple classes, you have to use a multinomial regression with softmax, make sures that the probability among the class prediction equals out to 100, the highest probability will be the predictive class

Multiple class model:
p(y=2|xi)= (1+e^-(b12*xi1+...+B02)/ sum of 1+e^-(B1k * xi + ... + B0k))

Loss function:
Loss(yhat,y)= sum of -[sum of (yi)log(yhati^k)] /  # of elements

Gradient:
dloss/dB1^k = -sum[yi - yhati^k]x1i

5. Optimization
Batch Gradient Descent: Loss and average of all the training examples and finding the gradient and updating the parameters
Stochastic Gradient Descent: Pick random training example, find its updated loss and then update - very slow to converge
Mini Batch: Take a small batch of the data and perform gradient descent, so there is less noise and its faster

Over fitting happens when a weight for a parameter becomes way too large. In this case we can use Regularization, which is basically adding an extra term

Regularization Terms
L2, Ridge, Gaussian 
L1, Lasso, Laplace 

Best tuned through cross validation

Early stopping parameters to prevent over fitting

Mix Max Scaling
XiScaled = (Xi-Ximin)/Ximax-Ximin

6. Performance
McFadden's pseudo-R^2
.2-.4 good fit

naive bayes - generative
logistic regression - discriminative 

### Support Vector Machine
Cares more about how correctly things are classified than the exact probability of classification

Instead of using a decision threshold, it cares more about a decision boundary

Decision boundary can be defined by this formula: 0=w*x-b
Positive examples: 1=w*x-b
Negative examples: -1=w*x-b
These formulas are dot products

The distance between the negative and positive example line is called a margin and can be defined by 2/[norm](https://www.varsitytutors.com/linear_algebra-help/norms) of w ||w||

Goal is maximize the margin without going past the negative and positive examples, we need to minimize the norm of W

w^tx-b >= 1 when yi=1
w^tx-b <= 1 when yi=-
yi(w^tx-b)>=1 -> constraints
min ||w|| -> optimize

constrained optimization problem

quadratic programming to fit model weights would result in hard margin SVMs

soft margin svms
min ||w||^2 + c(regularization) sum of ei (error term)
yi(w^tx-b)>=1-ei

we can max our error term like so:
ei=max(0,1-yi(w^tx-b))

we can turn this into an optimization formula with no constraints
min ||w||^2 + regularization of sum of *max (0,1-yi(w^tx-6))

*hinge loss

here we can plot it and also use sub gradient descent with pegasos

Kernel Trick -> Allows us to avoid transforming out feature to larger dimensions and still get dot product

Cant use the kernel trick with svm in a primal form -> dual of svms

representor theorem to represent the weights of svms 
w=sum of ai* yi *xi

take the dual
max sum of ai - 1/2 * sum of aj * ak * yj * yk* xj^Txk
max sum of ai - 1/2 * sum of aj * ak * yj * yk* kernel(could be linear, polynomial, gaussian) function of(xj^Txk)

great for # example N is much larger than the dimensions we want to project

RBF Kernel (Gaussian kernel)
Krbf = e^(-(||x-xi||^2)/2sigma ^2)

if sigma is too small, overfitting
if sigma is too large, underfitting

SVMS can be used for regression

Key Takeaway 
linearly separate separable data
Hard SVM v Soft SVM
Soft SVM has slack variables 
Soft SVM can use soft gradient descent to optimize
We can add feature interaction term which is preferred when we have low dimension 
If we dont have large data, and we want to project into a high dimensional space, we can use the kernel trick
SVM, are distance based with margins, if there is a low number of examples,start with linear svm
If you have a ton a points but not that many features you might want to start with logistic regression

## Unsupervised Learning
### K-Means
Organize data in K # of groups
Pick centroid and find all distance and then the data point will assume the centroid label when its the closest
Then update centroid to be the average of the cluster
Then repeat
When the centroid doesn't change then it has converged
We take the euclidean distance 
We can sum the distance and get the inertia
We want to minimize the inertia to form the best cluster
We need to find all possible clusters and measure the inertia and compare
Time complexity n^(kd+1)
with lloyd's algorithm we can get nkd
We're approximating the best cluster

When there is local minima we can get stuck there

Bisected K-means
whatever cluster has more inertia then we perform kmeans on that cluster

1. Optimize
Use elbow method, we plot all clusters and map the inertia
Silhouette Method
a(i)=avg(d(i,j)) 
s(i)=(b(i)-a(i)/max(a(i),b(i)))
if we plot avg the si per cluster, we get a peak and that is the silhouette coefficient

Centroid Classifier
K-medians
    - manhattan distance
K-medoids
    - k cluster average, than use the real data point closest
K-modes
    - gauges differences similarities 

Single linkage cluster
takes closest point in each cluster
and then compare them to the other parts of clusters
complete linkage cluster takes the farthest points and compare them

K means is distance based and perform scaling and standardize
xhat = x-xavg / sigma
 KD tree help performance
 Vulnerable to high dimensional data
 

### Singular Value Decomposition


input array can solve for three different matrices 

A=UEV^T
U - Rotation
Sigma - Scaling
V - Final rotation

A1 -> rotates axis, maps the points on the axis and rotates it back
Another rotation is performed perpendicular to the first axis

Go through Rank 1 to Rank N to find the best coverage
We can only use m-n rows for this system
in the second matrix there is you take the first element of the and then divide the sum of the whole matrix and that will give you the variance percentage

plot the coverage of information for each rank and you want to find the ranks that will cover 95% of the 

These vectors comes from Eigendecomposition

A=UEV^T <- each point is an eigen value
A = (ui=(Avi/sigma)(eigenvalues AA^T)(eigenvectors (A^T)A))

Eigendecomposition can only really work on square matrices 
The ranks of the Eigen values are not perpendicular so we cant break them down

PCA - Principal component analysis
A is standardized

(A^T)A/N-1 = VLV^T
(A^T)A -> usually unstable

A standardized =(UE) principal components V^T

We can use PCA or SVD for dimensionality reduction
Also nonlinear dimension reduction like Kernel PCA

## Deep Learning
### Neural Networks
Takes the input and multiple by the weight, then added up to summation then passed through a nonlinear function sigmoid function(activation) then result into output. sigma(w transpose x) = predictive output then we can add a bias inside of the (w transpose x + 1*bias)

We use the same loss like we do in logistic regression and then take the gradient 

Then we can update the weights

The output will be feed into another logistic regression

Each neuron will have to make a decision boundary so it can have a full output 

How complex the data is the depth of the network is greater

OPTIMIZATION
Loss function of cross entropy

To find the deviation of these function you use numerical gradient - takes a long time
Analytic Gradient -> take d of loss 
The chain rule, to know the partial d, we can do:
dL/dw6 = dL/dYOut * dYOut/dYIn * dYIn/dw6

We can use chain rule and dynamic programming to do back propagation 

Training we need a forward pass
and when we have the loss we can do back propagation to update the weights
Using the same algorithm

To optimize we can use stochastic gradient descent, but slow
We can use momentum into gradient descent 
w^t+1 = w^t - rDelta^tLoss - rDelta^t-1Loss, the momentum term keys to the algorithm how much the past should affect the current adjustment
w^t+1 = w^t - v^t
v^t = yv^t-1 - rDeltaLoss
Problem with momentum, is that you can skip over global optima

Adagrad can be used instead momentum
which utilizes a learning rate for a parameter
w1^t+1=w1^t - r1^t * (dloss^t/dw1)
r1^t = rgeneral/sqrt((dloss^t-1/dw1)^2 + ... + (dloss^t-n/dw1)^2) + e

Adam
combines momentum with an adoptive learning rate
w1^t+1=w1^t - (rgeneral/sqrt(vhat)+e) * mhatt
mt = b1 * mt-1 + (1-b) deltaLoss^t
vt = b1 * vt-1 + (1-b) (deltaLoss^t)^2
b= beta hyper parameters 
One instance
mthat = mt/1-beta1^t
vthat = vt/1-beta2^t

Other optimizers 
adaelta
rmsprop

Two major problems occur in optimization 
1. exploding gradients where values become to big
2. vanish gradients where values become too small

INITIALIZATION
To solve this we can use
initialization where we start each edge with a weight
bad way, is to initialize with uniformed or normal distribution
Xavier / Glorot(glowrow) initialization
better way is to initialize with normal where mean is o and sigma is sqrt2/fi+f0
fi = number of inputs or nodes in the layer
f0 = number of output nodes in the next layer

activation functions
symmetrical and rectified linear unit - all neg value will b 0 and all positive value will be positive
pro
- more computationally efficient
- tends to produce better model performance
- sparsity (reduces over fitting)
    - 0 value for all negative inputs

con
- uncapped activation, exploding gradient
- dying ReLU problem
    - once the neuron is zero its zero FOREVER

Kaiming for asymmetric activation functions like leaky reLU
initialize with normal where mean is o and sigma is sqrt2/fi

Feature scaling

Activation Function
Sigmoid
ReLU variants - start with frist
Hyperbolic tangent tanh
Can use different activation function in each layer

To output multiple class probability - softmax
for regression - use linear activation function

Performance Checking
Loss Functions
Mean Sq Error L2
L(y,yhat) = sum (yi-yihat)^2 / n
Mean absolute error L1
L(y,yhat) = sum |yi-yihat| / n
cross-entropy log loss
L(y,yhat) = (ylog(yhati))+(1-y)log(1-yhat)

Regularization 
L1/L2 added to loss
L1
L(y,yhat) = -sum yi log (yhati) + lambda sum|wi|
L2
L(y,yhat) = -sum yi log (yhati) + lambda sum wi^2
Dropout
When neural networks node has constantly updating probability of success, if nodes don't meet that probability requirement they are dropped
Cons 
When forming predications outside, nodes wont be dropped out and can mess things up
So we use inverted drop out, we divide each layer by probability, so it will match on prediction and test time

Architecture
start with one hidden layer 
#neurons = input+output / 2

start with more layers and unit than u need
see which weights are near zero after training then prune

Feature Scaling
Activation Functions
Loss Function 
Optimizers
Regularization Terms
Dropout Specifications
Architecture 

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
### Line of Best Fit
The line through data points which best describe the relationship of a dependent variable with one or more independent variables. Ordinary least squares can be used to find the line of best fit
### P Value
The probability of finding a particular result or a greater result given a null hypothesis being true
### Confidence Interval
The possible range of unknown value. Often comes with some degree of probability 
### Correlation
The relationship between a dependent and independent variable
### R Squared
Also the coefficient of determination the percentage of variance in the dependent variable explained by the independent variables
### Residuals 
The distance between points and a particular line
### Independent Variable
A variable whose variation is independent from other variables
### One-hot Encoding
An encoding for categorical variable where every value that a variable can take on is represented as a binary vector
### Dependent Variable
A variable whose variation depends on other variables
### Variance Inflation Factor
A measure of multicollinearity in a regression model
### Collinearity
When one or more multicollinearity independent variables are not actually independent
### Nonlineaer Regression
A type of regression which models nonlinear relationships in the independent variables
### Simpson's paradox
When a pattern emerges in segments of examples but is no longer present when the segments are grouped together
### Statsmodels
Python module which provides various statistical tools
### Coefficient 
Another name for a parameter int he regression model
### Sigmoid FUnction 
also the logistic function, a function which outputs a range from 0 to 1
### Closed Form Solution
For our case, this is what ordinary least squares provides for linear regression. Its a formula which solves an equation
### Cross-Entropy Loss
A loss function which is used in classification task. It's technically the entropy of true labels plus the KL Divergence of the predicted and true labels. Minimizing the the entropy minimizes the difference between the true and predicted label distributions
### Parameters 
Also weights or coefficients. Values to be learned during the model training.
### Learning Rate
A multiple typically less than 1, used during the parameter update step during model training to smooth the learning process
### Odds Ratio
The degree of associate between two events. If the odds ratio is 1, then the two events are independent. If the odds ratio is greater than 1, the events are positively correlated, otherwise the events are negatively correlated
### Multinomial Logistic Regression 
Logistic Regression in which there are more than two classes to be predicted across
### Softmax
A sigmoid which is generalized to more than two classes to be predicted against
### Gradient Descent
An iterative algorithm with the goal of optimizing some parameters of given function with respect to some loss function. If done in batches, all of the examples are considered for an iteration of gradient descent. In mini-bath gradient descent, a subset of examples are considered for a single iteration. Stochastic gradient descent considers a single example per gradient descent iteration.
### Downsampling
Removing a number of majority class examples. Typically done in addition to upweighting
### Upweighting
Increasing the impact a minority class example has on the loss function. Typically done in addition to downsampling.
### Epoch 
One complete cycle of training on all the examples 
### Regularization
A technique of limiting the ability for a model to overfit by encouraging the values parameter to take on smaller values
### Early Stopping
Halting the gradient descent process prior to approaching the minima or maxima
### Mcfadden's Pseudo R-Squared
An analog to linear regression's R-squared which typically takes on smaller values than the traditional R-Squared
### Generative Model
A model which aims to approximate the joint probability of the features and labels
### Discriminative Model
A model which aims to approximate the conditional probability of the features and labels
### Support Vectors
The most difficult to separate points in regards to decision boundary. They influence the location and orientation of the hyperplane
### Hyperplane
A decision boundary in any dimension
### Norm
Here, the L2 norm, is the sq root of the sum of squares of each element in a vector
### Outlier
A feature of group of features which vary significantly from the other features
### Slack
The relaxing of the constraint that all examples must lie outside of the margin. This creates a soft-margin SVM
### Hinge Loss
A loss function which is used by a soft-margin SVM
### Sub-gradient
THe gradient of a non differentiable function
### Non-differentiable
A function which has kinks in which a derivative is not defined
### Convex Function
Function with one optima
### Kernel Trick
The process of finding the dot product of a high dimensional representation of feature without computing the high dimensional representation itself. A common kernel is the radial basis function kernel
### Centroid
The location of center of a cluster in n-dimensions
### Inertia
The sum of distances between each point and the centroid
### Local Optima
A maxima or minima which is not the global optima
### Non-convex function
A function which has two or more instances of zero-slope
### Elbow Method
A method of finding the best value for k in k-means. It involves finding the elbow of the plot of range of ks and their respective inertias
### Silhouette Methods
A method of finding the best value for k in k-means. It takes into account the ratios of the inter and intra clusters distances
### K-means++ 
Using a weighted probability distribution as a way to find the initial centroid locations for the k-means algorithm
### Agglomerative Clustering
A clustering algorithm that builds a hierarchy of sub clusters that gradually group into a single cluster. Some techniques for measuring distances between clusters are single-linkage and complete-linkage methods
### Singular Value Decomposition
Also SVD, a process which decomposes a matrix into a rotation and scaling terms. Its is a generalization of eigendecomposition
### Rank r Approximation
Using up to and including, the rth terms in the singular value decomposition to approximate an orginal matrix
### Dimensionality Reduction
The process of reducing the dimensionality of features. This is typically useful to speed up the training of models and in some cases, allow for a wider number of machine learning algorithms to be used on the examples. This can be done with SVD or PCA and as well certain types of neural networks such as autoencoders
### Eigendecomposition
Applicable only to square matrices, the method of factoring a matrix into its eigenvalues and eigenvectors. AN eigenvector is a vector which applies a linear transformation to some matrix being factored. THe eigenvalues scale the eigenvector values
### Principal Component Analysis
Also PCA, is eigendecomposition performed on the covariance matrix of some particular data. The eigenvectors then describe the principle components and the eigenvalues indicate the variance described by each principals components. Typically, assumption is not true, then you can use kernel PCA
### Orthogonal
Perpendicular is n-dimensions
### Neuron
Sometimes called a perceptron, a neuron is a graphical representation of the smallest part of a neural network. For the Machine Learning Crash Course reference neurons as nodes or units
### Gradient
A vector of partial derivatives. In terms of neural networks, we often use the analytical gradient in software and use the numerical gradient as a gradient checking mechanism to ensure the analytics gradient is accurate
### Parameter 
Any train value in a model,
### Feature Transformation
A mathematical function applied to features
### Hidden Layer
A layer that's not the input or output layer in a neural network
### Backpropagation
The use of the derivative chain rule along with dynamic programming to determine the gradients of the loss function in neural networks
### Forward Pass
Calculating an output of a neural network for a particular input
### Local Optima 
A maxima or minima which is not the global optima
### Momentum
A concept applied to gradient descent in which the gradients applied to the weight updates depends on previous gradients
### Adagrad
An optimizer used to update the weights of a neural network in which different learning rates are applied to different weights
### Adam
A common gradient descent optimizer that takes advantage of momentum and adaptive learning rates
### Hyperparameter
Any parameter associated with a model which is not learned
### Optimizers
Techniques which attempt to optimize gradient descent 
### Vanishing Gradient
The repeated multiplication of small gradients resulting in an overflow or 0 value products
### Exploding Gradient
The repeated multiplication of large gradients resulting in an overflow or infinity value products
### Initialization Techniques
Ways to cleverly initialize the weights of neural networks in an attempt avoid vanishing and exploding gradients. Kaiming initialization used with asymmetric activation functions and xavier glorot initialization, used with symmetric activation function are both examples. These techniques usually depend on the fan in and fan out per layer
### Activation Function
The function used to the output of a neuron. These activations can be linear or nonlinear. If they're nonlinear, they can be symmetric or asymmetric
### Rectified Linear Unit
An asymmetric activation function which outputs the value of the positive inputs and zero otherwise. There are variations such as the Leaky ReLU. They can be susceptible to the dead neuron problem but generally perform well in practice
### Hyperbolic Tangent
A symmetric activation function which ranges from -1 to 1
### L2 Loss
The sum of the squared errors of all training examples. Not to be confused with L2 regularization
### Mean Absolute Error
The average of the absolute differences across the training examples
### Dropout
A regularization technique used per layer to reduce over fitting. Dropout involves randomly omitting neurons from the neural network structure at each training iteration. Effectively, dropout produces an ensemble of neural networks. Dropout is incomplete without adjusting for the dropout in preparation of prediction
### Binary Classification
A supervised learning task in which there are two possible outputs
### Pruning Neurons
Removing neurons from a neural network in an effort to reduce the number of model parameters if by removing the neurons equivalent performance can be obtained