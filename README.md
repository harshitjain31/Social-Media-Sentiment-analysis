# Tweet Sentiment Analysis

This project involves developing a Machine Learning model to detect the sentiment associated with tweets, classifying them as either positive or negative. Various Natural Language Processing (NLP) techniques and machine learning models are used to achieve accurate sentiment classification.

## Introduction

The goal of this project is to build a sentiment analysis model that classifies tweets as positive or negative. I experimented with several models to find the best-performing one for our dataset.

## Technologies Used

- **Programming Languages:** Python
- **Libraries & Frameworks:**
  - **Data Handling and Visualization:**
    - numpy
    - pandas
    - seaborn
    - matplotlib
    - wordcloud
    - PIL
  - **Machine Learning & NLP:**
    - scikit-learn
    - torch
    - nltk
- **Tools:**
  - Jupyter Notebook
  - Git

## Data Preparation

1. **Data Collection:** Collected a dataset of tweets.
2. **Text Preprocessing:** Included tokenization, stop-word removal, and lemmatization.
3. **Feature Extraction:** Used techniques like CountVectorizer and TfidfVectorizer to convert text data into numerical features.

## Model Training

We evaluated the following models to determine which best fits our dataset:

- **Logistic Regression**
- **Decision Trees**
- **Random Forest Classifier**
- **Naive Bayes**
- **AdaBoost**

## Results

The Random Forest model was selected as the best-performing model based on the F1 score of 0.718067.


