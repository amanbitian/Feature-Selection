# Feature-Selection

![1_H7XDAYHFWIy4JXy-WXhUZw](https://user-images.githubusercontent.com/86042628/140170825-ecada75f-e199-44a0-a60f-4c33ca7428c3.jpeg)

![image](https://user-images.githubusercontent.com/86042628/140192477-69f88401-69e4-4d48-b462-9d6465310ee8.png)


I will share 3 filter (Feature selection) techniques that are easy to use and also gives good results.
1. Univariate Selection
2. Feature Importance
3.Correlation Matrix with Heatmap

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

