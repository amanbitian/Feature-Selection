# Feature-Selection

## What is Feature selection?
As the name suggests, `it is a process of selecting the most significant and relevant features from a vast set of features in the given dataset.`

For a dataset with d input features, the feature selection process results in k features such that k < d, where k is the smallest set of significant and relevant features.

So feature selection helps in finding the smallest set of features which results in

    * Training a machine learning algorithm faster.
    * Reducing the complexity of a model and making it easier to interpret.
    * Building a sensible model with better prediction power.
    * Reducing over-fitting by selecting the right set of features.

### Methods of **feature selection**

![1_H7XDAYHFWIy4JXy-WXhUZw](https://user-images.githubusercontent.com/86042628/140170825-ecada75f-e199-44a0-a60f-4c33ca7428c3.jpeg)

![image](https://user-images.githubusercontent.com/86042628/140192477-69f88401-69e4-4d48-b462-9d6465310ee8.png)


I will share 3 filter (Feature selection) techniques that are easy to use and also gives good results.
1. Univariate Selection
2. Feature Importance
3. Correlation Matrix with Heatmap

## 1. Univariate Selection
Statistical tests can be used to select those features that have the strongest relationship with the output variable.
The scikit-learn library provides the SelectKBest class that can be used with a suite of different statistical tests to select a specific number of features.
The example below uses the chi-squared (chi²) statistical test for non-negative features to select 10 of the best features from the Mobile Price Range Prediction Dataset.

`import pandas as pd`          
`import numpy as np`  
`from sklearn.feature_selection import SelectKBest`  
`from sklearn.feature_selection import chi2`  
`data = pd.read_csv("D://Blogs//train.csv")`  
`X = data.iloc[:,0:20]  #independent columns`  
`y = data.iloc[:,-1]    #target column i.e price range`  
`#apply SelectKBest class to extract top 10 best features`  
`bestfeatures = SelectKBest(score_func=chi2, k=10)`  
`fit = bestfeatures.fit(X,y)`  
`dfscores = pd.DataFrame(fit.scores_)`  
`dfcolumns = pd.DataFrame(X.columns)`  
`#concat two dataframes for better visualization`   
`featureScores = pd.concat([dfcolumns,dfscores],axis=1)`  
`featureScores.columns = ['Specs','Score']  #naming the dataframe columns`  
`print(featureScores.nlargest(10,'Score'))  #print 10 best features` 

## 2. Feature Importance
You can get the feature importance of each feature of your dataset by using the feature importance property of the model.
Feature importance gives you a score for each feature of your data, the higher the score more important or relevant is the feature towards your output variable.
Feature importance is an inbuilt class that comes with Tree Based Classifiers, we will be using Extra Tree Classifier for extracting the top 10 features for the dataset.

`import pandas as pd`    
`import numpy as np`    
`data = pd.read_csv("D://Blogs//train.csv")`  
`X = data.iloc[:,0:20]  #independent columns`  
`y = data.iloc[:,-1]    #target column i.e price range`  
`from sklearn.ensemble import ExtraTreesClassifier`  
`import matplotlib.pyplot as plt`  
`model = ExtraTreesClassifier()`  
`model.fit(X,y)`  
`print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers`  
`#plot graph of feature importances for better visualization`  
`feat_importances = pd.Series(model.feature_importances_, index=X.columns)`  
`feat_importances.nlargest(10).plot(kind='barh')`  
`plt.show()`  

## 3.Correlation Matrix with Heatmap
Correlation states how the features are related to each other or the target variable.
Correlation can be positive (increase in one value of feature increases the value of the target variable) or negative (increase in one value of feature decreases the value of the target variable)
Heatmap makes it easy to identify which features are most related to the target variable, we will plot heatmap of correlated features using the seaborn library.

`import pandas as pd`  
`import numpy as np`  
`import seaborn as sns`  
`data = pd.read_csv("D://Blogs//train.csv")`  
`X = data.iloc[:,0:20]  #independent columns`  
`y = data.iloc[:,-1]    #target column i.e price range`  
`#get correlations of each features in dataset`  
`corrmat = data.corr()`  
`top_corr_features = corrmat.index`  
`plt.figure(figsize=(20,20))`  
`#plot heat map`  
`g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")`  

# Wrapper methods
In wrapper methods, the feature selection process is based on a specific machine learning algorithm that we are trying to fit on a given dataset.
It follows a **greedy search approach** `by evaluating all the possible combinations of features against the evaluation criterion`. *The evaluation criterion is simply the performance measure which depends on the type of problem,* 
for e.g. For regression evaluation criterion can be `p-values, R-squared, Adjusted R-squared`,   
similarly for classification the evaluation criterion can be `accuracy, precision, recall, f1-score`, etc. Finally, it selects the combination of features that gives the optimal results for the specified machine learning algorithm.
Flow Chart: wrapper feature selection

![46072IMAGE2](https://user-images.githubusercontent.com/86042628/140195491-031a3600-cd6b-487b-a628-ba0083c3c98b.gif)

Most commonly used techniques under wrapper methods are:

    1. Forward selection
    2. Backward elimination
    3. Bi-directional elimination(Stepwise Selection)
   
## 1. Forward selection

inshort: ![forward selection](https://user-images.githubusercontent.com/86042628/140197305-aa5a9769-600b-41d9-b555-89d7b17cbe09.PNG)


In forward selection, *we start with a null model and then start fitting the model with each individual feature one at a time and select the feature with the minimum p-value. Now fit a model with two features by trying combinations of the earlier selected feature with all other remaining features. Again select the feature with the minimum p-value. Now fit a model with three features by trying combinations of two previously selected features with other remaining features. Repeat this process until we have a set of selected features with a p-value of individual features less than the significance level.*

In short, the steps for the forward selection technique are as follows :

    * Choose a significance level (e.g. SL = 0.05 with a 95% confidence). 
    * Fit all possible simple regression models by considering one feature at a time. Total ’n’ models are possible. Select the feature with the lowest p-value.
    * Fit all possible models with one extra feature added to the previously selected feature(s).
    * Again, select the feature with a minimum p-value. if p_value < significance level then go to Step 3, otherwise terminate the process.

## 2. Backward Feature Selection
![Backward selection](https://user-images.githubusercontent.com/86042628/140198879-097db5a2-f8f2-484d-b6fe-c7ee7543ffa6.PNG)

In backward elimination, ***we start with the full model (including all the independent variables) and then remove the insignificant feature with the highest p-value(> significance level). This process repeats again and again until we have the final set of significant features.***

In short, the steps involved in backward elimination are as follows:

    * Choose a significance level (e.g. SL = 0.05 with a 95% confidence).
    * Fit a full model including all the features.
    * Consider the feature with the highest p-value. If the p-value > significance level then go to Step 4, otherwise terminate the process.
    * Remove the feature which is under consideration.
    * Fit a model without this feature. Repeat the entire process from Step 3.


## 3. Bi-directional elimination(Step-wise Selection)

It is similar to forward selection but the difference is while adding a new feature it also checks the significance of already added features and if it finds any of the already selected features insignificant then it simply removes that particular feature through backward elimination.

Hence, **It is a combination of forward selection and backward elimination.**

In short, the steps involved in bi-directional elimination are as follows:

    * Choose a significance level to enter and exit the model (e.g. SL_in = 0.05 and SL_out = 0.05 with 95% confidence).
    * Perform the next step of forward selection (newly added feature must have p-value < SL_in to enter).
    * Perform all steps of backward elimination (any previously added feature with p-value>SL_out is ready to exit the model).
    * Repeat steps 2 and 3 until we get a final optimal set of features.


## Drawback of GREEDY SEQUENTIAL ALGORITHMS
• Extremely computationally expensive
• Often not feasible due to number of features in dataset
• Feature space optimised for a specific algorithm
• Should provide the highest performance

# Embedded feature selection methods
