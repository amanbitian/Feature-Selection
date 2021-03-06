[Source](https://www.analyticsvidhya.com/blog/2016/12/introduction-to-feature-selection-methods-with-an-example-or-how-to-select-the-right-variables/)

Table of Contents

    Importance of Feature Selection in Machine Learning
    Filter Methods
    Wrapper Methods
    Embedded Methods
    Difference between Filter and Wrapper methods
    Walkthrough example

 
## 1. Importance of Feature Selection in Machine Learning

Machine learning works on a simple rule – if you put garbage in, you will only get garbage to come out. By garbage here, I mean noise in data.

This becomes even more important when the number of features are very large. You need not use every feature at your disposal for creating an algorithm. You can assist your algorithm by feeding in only those features that are really important. I have myself witnessed feature subsets giving better results than complete set of feature for the same algorithm. Or as Rohan Rao puts it – “Sometimes, less is better!”

Not only in the competitions but this can be very useful in industrial applications as well. You not only reduce the training time and the evaluation time, you also have less things to worry about!

Top reasons to use feature selection are:

    It enables the machine learning algorithm to train faster.
    It reduces the complexity of a model and makes it easier to interpret.
    It improves the accuracy of a model if the right subset is chosen.
    It reduces overfitting.

Next, we’ll discuss various methodologies and techniques that you can use to subset your feature space and help your models perform better and efficiently. So, let’s get started.

 
## 2. Filter Methods

filter_1

Filter methods are generally used as a preprocessing step. The selection of features is independent of any machine learning algorithms. Instead, features are selected on the basis of their scores in various statistical tests for their correlation with the outcome variable. The correlation is a subjective term here. For basic guidance, you can refer to the following table for defining correlation co-efficients.

fs1

    Pearson’s Correlation: It is used as a measure for quantifying linear dependence between two continuous variables X and Y. Its value varies from -1 to +1. Pearson’s correlation is given as:

fs2

    LDA: Linear discriminant analysis is used to find a linear combination of features that characterizes or separates two or more classes (or levels) of a categorical variable.

    ANOVA: ANOVA stands for Analysis of variance. It is similar to LDA except for the fact that it is operated using one or more categorical independent features and one continuous dependent feature. It provides a statistical test of whether the means of several groups are equal or not.

    Chi-Square: It is a is a statistical test applied to the groups of categorical features to evaluate the likelihood of correlation or association between them using their frequency distribution.

One thing that should be kept in mind is that filter methods do not remove multicollinearity. So, you must deal with multicollinearity of features as well before training models for your data.

 
## 3. Wrapper Methods

wrapper_1

In wrapper methods, we try to use a subset of features and train a model using them. Based on the inferences that we draw from the previous model, we decide to add or remove features from your subset. The problem is essentially reduced to a search problem. These methods are usually computationally very expensive.

Some common examples of wrapper methods are forward feature selection, backward feature elimination, recursive feature elimination, etc.

    **Forward Selection:** Forward selection is an iterative method in which we start with having no feature in the model. In each iteration, we keep adding the feature which best improves our model till an addition of a new variable does not improve the performance of the model.
    **Backward Elimination:** In backward elimination, we start with all the features and removes the least significant feature at each iteration which improves the performance of the model. We repeat this until no improvement is observed on removal of features.
    **Recursive Feature elimination:** It is a greedy optimization algorithm which aims to find the best performing feature subset. It repeatedly creates models and keeps aside the best or the worst performing feature at each iteration. It constructs the next model with the left features until all the features are exhausted. It then ranks the features based on the order of their elimination.

One of the best ways for implementing feature selection with wrapper methods is to use Boruta package that finds the importance of a feature by creating shadow features.

It works in the following steps:

    Firstly, it adds randomness to the given data set by creating shuffled copies of all features (which are called shadow features).
    Then, it trains a random forest classifier on the extended data set and applies a feature importance measure (the default is Mean Decrease Accuracy) to evaluate the importance of each feature where higher means more important.
    At every iteration, it checks whether a real feature has a higher importance than the best of its shadow features (i.e. whether the feature has a higher Z-score than the maximum Z-score of its shadow features) and constantly removes features which are deemed highly unimportant.
    Finally, the algorithm stops either when all features get confirmed or rejected or it reaches a specified limit of random forest runs.

For more information on the implementation of Boruta package, you can refer to this article :

For the implementation of Boruta in python, refer can refer to this article.

 
## 4. Embedded Methods

embedded_1

Embedded methods combine the qualities’ of filter and wrapper methods. It’s implemented by algorithms that have their own built-in feature selection methods.

Some of the most popular examples of these methods are LASSO and RIDGE regression which have inbuilt penalization functions to reduce overfitting.

    **Lasso regression performs L1 regularization** which adds penalty equivalent to absolute value of the magnitude of coefficients.
    **Ridge regression performs L2** regularization which adds penalty equivalent to square of the magnitude of coefficients.

For more details and implementation of LASSO and RIDGE regression, you can refer to this article.

Other examples of embedded methods are Regularized trees, Memetic algorithm, Random multinomial logit.
## 5. Difference between Filter and Wrapper methods

The main differences between the filter and wrapper methods for feature selection are:

    Filter methods measure the relevance of features by their correlation with dependent variable while wrapper methods measure the usefulness of a subset of feature by actually training a model on it.
    Filter methods are much faster compared to wrapper methods as they do not involve training the models. On the other hand, wrapper methods are computationally very expensive as well.
    Filter methods use statistical methods for evaluation of a subset of features while wrapper methods use cross validation.
    Filter methods might fail to find the best subset of features in many occasions but wrapper methods can always provide the best subset of features.
    Using the subset of features from the wrapper methods make the model more prone to overfitting as compared to using subset of features from the filter methods.

 
## 6. Walkthrough example

Let’s use wrapper methods for feature selection and see whether we can improve the accuracy of our model by using an intelligently selected subset of features instead of using every feature at our disposal.

We’ll be using stock prediction data in which we’ll predict whether the stock will go up or down based on 100 predictors in R. This dataset contains 100 independent variables from X1 to X100 representing profile of a stock and one outcome variable Y with two levels : 1 for rise in stock price and -1 for drop in stock price.

To download the dataset, click here.

Let’s start with applying random forest for all the features on the dataset first.

library('Metrics')
