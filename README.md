# Project 3: Reddit Scraping
#### Sara Soueidan /// 26.04.2020
__________________________________________________________________________________________________

## Problem Statement

It is possible to determine which subreddit a specific reddit post was posted. Using an array of classification models, we found that deciding whether or not a post belonged to r/Cooking or r/Baking depended of a few fundamental decisions in the iterative data science process. Principally, we explored whether a logistic regression classification model, Naive Bayes or Decision Tree classification model was better at predicting the a post's subreddit. A successful model with be able to predict with a high-degree of accuracy (Accuracy > 95%) a post's origin subreddit.

## Executive Summary

### Data Collection

**Notebook 00**
To begin, we scraped the Reddit r/Cooking and r/Baking using the Pushshift API to obtain the 100,000 most recent posts from each subreddit using Python's `requests` library. We pulled six features from each post (title, body, datetime posted, post ID, subreddit and media presence).

### Data Cleaning & EDA

**Notebooks 01 and 02**
Standard EDA techniques were applied to the raw data (missing, value counts, descriptive statistics). The df_cooking and df_baking were then combined into one df_reddit_cleaned to be used for the rest of the project. The df_reddit_cleaned DataFrame was then processed and cleaned. Cleaning occurred throughout all stages of this project. Rows with Null values in either the Title or Selftext columns were dropped. To begin processing the data, a tokenizer was applied to both the title and selftext columns to assess frequency of words in the corpus. Using these results, the stopword list was modified to include words found in both the top 20 of Cooking and Baking. The data was then processed for both Title and Selftext. Ultimately, Title was chosen as the principle component of the X variable for models (Selftext was orders of magnitude larger than Title, making it computationally intense). The column Subreddit was also converted to binary values. The processed df_reddit_cleaned was exported as reddit_processed.csv for use in model building.

### Data Preprocessing & Modeling
Each model run resulted in an output specifying model specific values (estimator, transformer, train score, test score, best parameters and in the case of logistic regression: coefficient dictionary). Each model was run a minimum of 5 times.


**Notebook 03: Logistic Regression**

The first model utilizes a classic logistic regression with a CountVectorized or TfidfVectorized X variable (post title). 
These models were run using GridSearchCV with a large array of parameters,  further run with hypertuned parameters.


**Notebook 04: Decision Tree**

The second model utilizes a decision tree with a CountVectorized or TfidfVectorized transformed X variable (post title). 
These models were run using GridSearchCV with a large array of parameters,  further run with hypertuned parameters.


**Notebook 05: Naive Bayes**

The third model (and final model) evaluated utilizes a Naive Bayes classifier with a CountVectorized or TfidfVectorized transformed X variable (post title).
These models were run using GridSearchCV with a large array of parameters,  further run with hypertuned parameters.

### Model Evaluation

**Notebook 06**
Comparing 

## Data Dictionaries

**Data Dictionary for Input Data**
- df_cooking
- df_baking
- df_reddit
- df_reddit_cleaned
- df_reddit_process

| Feature     | Meaning                          | Type   |
|-------------|----------------------------------|--------|
| title       | reddit post title                | object |
| created_utc | epoch when post was submitted    | int64  |
| selftext    | reddit post body                 | object |
| subreddit   | name of subreddit                | object |
| id          | unique id for post               | object |
| media_only  | boolean for has media attachment | object |


**Data Dictionary for Score Dictionary**

| Feature     | Meaning                          | Type   |
|-------------|----------------------------------|--------|
| model       | model name                       | object |
| transformer | transformer name                 | object |
| train score | training score                   | float64|
| test score  | testing score                    | float64|

## Code Notebooks

### [Data Collection]('./00_import_data.ipynb')

- Import Libraries
- Get Data
- Make DataFrame
- Save to CSV ('cooking.csv' and 'baking.csv')

### [Data Exploration and Cleaning]('./01_clean_data.ipynb')

- Import Libraries
- Read in Data
- Combine Cooking and Baking DataFrames
- Remove Nulls (Missing and Duplicates)
- Save to CSV ('reddit_cleaned.csv')

### [Data Processing and Cleaning]('./02_process_data.ipynb')

- Import Libraries
- Read in Data
- Drop Nulls (Again)
- Review Top N Words in Corpus
- Determine Additional Stopwords
- Remove Stopwords
- Process Words
- Convert Subreddit Column to Binary
- Save to CSV ('reddit_processed.csv')

### [Data Modeling: Logistic Regression]('./03_model_data_logistic.ipynb')

- Import Libraries
- Read in Data
- Check for NaNs / Drop NaNs in X, y Features
- Select X and y Features (Title and Subreddit)
- Train Test Split
- Build Estimator Function
- Run Models (Complex -> Simple)
    - Save best results dictionary to scores dataframe

### [Data Modeling: Decision Tree]('./04_model_data_decision_tree.ipynb')

- Import Libraries
- Read in Data
- Check for NaNs / Drop NaNs in X, y Features
- Select X and y Features (Title and Subreddit)
- Train Test Split
- Build Estimator Function
- Run Models (Complex -> Simple)
    - Save best results to dictionary to scores dataframe

### [Data Modeling: Naive Bayes]('./05_model_data_naive_bayes.ipynb')

- Import Libraries
- Read in Data
- Check for NaNs / Drop NaNs in X, y Features
- Select X and y Features (Title and Subreddit)
- Train Test Split
- Build Estimator Function
- Run Models (Complex -> Simple)
    - Save best results to dictionary to scores dataframe

### [Model Evaluation]('./06_model_evaluation.ipynb')

- ROC AUC Scores
- Word Histograms
- Bag of Word Analysis
- Confusion Matrix

### Main Directory
Jupyter Notebooks (00 - 07)
[README] ('./README.md')
[Reddit Classification] ('./Reddit Classification.pdf')
[Sara Script] ('./sara.py')

### Data Folder
[Baking CSV]('./data/baking.csv')
[Cooking CSV] ('./data/cooking.csv')
[Reddit Cleaned CSV] ('./data/reddit_cleaned.csv')
[Reddit Processed CSV] ('./data/reddit_processed.csv')
[All Scores CSV] ('./data/all_scores.csv')

### Misc Folder
[Scratchpad] ('./misc/scratchpad.ipynb')
[Requirements] ('./misc/requirements.txt')
[starter_README] ('./misc/starter_README.md')

## Conclusion

It is, indeed, possible to predict from which subreddit a post belongs. From the [scores]('./data/all_scores.csv') it is possible to see that the best performing model was the K-NN model in combination with a CountVectorizer.


## To Do List

- add more viz
- delete extraneous libraries from notebooks
- build output csvs for model scores
- build output csv for all scores
- build out NB 07



