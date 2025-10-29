Sentiment Analysis on Movie Reviews (Enhanced)
Project Overview

This project performs sentiment analysis on movie reviews using a combination of rule-based and machine learning techniques. It predicts both the sentiment (Positive or Negative) and a rating out of 10 for each movie based on the textual feedback provided in the dataset. Additionally, the project provides visualizations to understand overall sentiment distribution and individual movie ratings.

Features

Data Preprocessing

Tokenization of review text.

Conversion to lowercase.

Removal of punctuation and non-alphabetic tokens.

Stopwords removal.

Lemmatization for standardizing words.

Sentiment Labeling

Rule-based labeling using predefined lists of positive and negative words.

Random assignment for neutral reviews to maintain balanced data.

Machine Learning Model

TF-IDF vectorization of cleaned text.

Logistic Regression model for sentiment classification.

Train-test split with model evaluation (accuracy and classification report).

Rating Prediction

Converts predicted sentiment probabilities into a 1–10 rating scale.

Assigns both predicted sentiment and predicted rating to all movies in the dataset.

Visualizations

Distribution of predicted ratings for all movies.

Overall predicted sentiment distribution.

Movie-specific sentiment pie chart visualization based on predicted ratings.

Usage

Place your dataset file (AI_dataset.csv) in the project directory. The dataset should have the following columns:

movie_name

feedback1, feedback2, feedback3, feedback4 (textual reviews)

Run the script:

python main.py


After the model is trained, the program will:

Display the predicted sentiment and rating for all movies.

Save the results to Predicted_Sentiments_and_Ratings.csv.

Generate overall visualizations for predicted ratings and sentiments.

Prompt the user to enter a movie name for a movie-specific sentiment pie chart.

Dependencies

Python 3.x

pandas

nltk

scikit-learn

matplotlib

seaborn

NLTK Data Downloads:

punkt

stopwords

wordnet

Example
Enter a movie name to visualize sentiment rating: Casablanca
Sentiment for 'Casablanca': Positive=3, Negative=1, Rating=7.7


The script will also display a pie chart showing the proportion of positive and negative sentiment for the entered movie.

Notes

The project uses a combination of rule-based and machine learning methods to handle small datasets effectively.

The predicted rating is calculated from the model’s probability output and scaled to 1–10.

This project can be extended to live user reviews or larger datasets for more accurate predictions.

If you want, I can also write a very short “project overview README” suitable for GitHub that’s under 200 words but still looks professional and descriptive.
