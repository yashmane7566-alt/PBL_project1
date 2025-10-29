# ===============================
# SENTIMENT ANALYSIS ON MOVIE REVIEWS (RATING-BASED)
# ===============================

import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import random
import matplotlib.pyplot as plt

# ===============================
# 1. Load Dataset
# ===============================
df = pd.read_csv("raw_dataset.csv.csv")
print("ORIGINAL DATASET\n")
print(df.info(), "\n")

# ===============================
# 2. Merge all feedback columns
# ===============================
df["all_feedback"] = (
    df["feedback1"] + " " +
    df["feedback2"] + " " +
    df["feedback3"] + " " +
    df["feedback4"]
)

# ===============================
# 3. Sentiment labeling
# ===============================
positive_words = [
    'good', 'great', 'amazing', 'excellent', 'love', 'fantastic', 'awesome', 'best',
    'enjoyed', 'wonderful', 'brilliant', 'superb', 'liked', 'masterpiece'
]
negative_words = [
    'bad', 'poor', 'terrible', 'hate', 'worst', 'boring', 'awful', 'disappointing',
    'mediocre', 'dull', 'forgettable', 'annoying', 'slow', 'uninteresting', 'messy'
]

def enhanced_sentiment(text):
    text = text.lower()
    pos = sum(word in text for word in positive_words)
    neg = sum(word in text for word in negative_words)
    if pos == 0 and neg == 0:
        return random.choice([0, 1])
    return 1 if pos > neg else 0

df['sentiment'] = df['all_feedback'].apply(enhanced_sentiment)
print("SENTIMENT LABELS GENERATED\n")
print(df[['movie_name', 'sentiment']].head(), "\n")

# ===============================
# 4. Text preprocessing
# ===============================
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [w for w in tokens if w.isalpha()]
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)

df["cleaned_text"] = df["all_feedback"].apply(preprocess)
print("\nCLEANED TEXT EXAMPLES\n")
print(df[["movie_name", "cleaned_text"]].head(), "\n")

# ===============================
# 5. TF-IDF
# ===============================
tfidf = TfidfVectorizer(max_features=500)
X = tfidf.fit_transform(df["cleaned_text"])
y = df["sentiment"]

# ===============================
# 6. Train-test split & model
# ===============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# ===============================
# 7. Evaluation
# ===============================
y_pred = model.predict(X_test)
print("MODEL EVALUATION\n")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# ===============================
# 8. Predictions & ratings
# ===============================
df['predicted_sentiment'] = model.predict(X)
df['predicted_sentiment'] = df['predicted_sentiment'].apply(lambda x: 'Positive' if x == 1 else 'Negative')

prob_pos = model.predict_proba(X)[:, 1]
df['predicted_rating'] = (prob_pos * 9 + 1).round(1)

print("\nFINAL DATASET WITH PREDICTED SENTIMENT AND RATING\n")
print(df[['movie_name', 'predicted_sentiment', 'predicted_rating']].head())

# ===============================
# 9. Save to CSV
# ===============================
df[['movie_name', 'predicted_sentiment', 'predicted_rating']].to_csv('Predicted_Sentiments_and_Ratings.csv', index=False)
print("\nResults saved to Predicted_Sentiments_and_Ratings.csv")

# ===============================
# 10. Simple visualizations
# ===============================
plt.figure(figsize=(6,4))
df['predicted_sentiment'].value_counts().plot(kind='bar', color=['lightgreen', 'salmon'])
plt.title('Predicted Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Number of Movies')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

plt.figure(figsize=(8,5))
plt.hist(df['predicted_rating'], bins=10, color='skyblue', edgecolor='black')
plt.title('Distribution of Predicted Movie Ratings (1–10)')
plt.xlabel('Predicted Rating')
plt.ylabel('Count')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# ===============================
# 11. Movie-specific chart (Rating-Based)
# ===============================
movie_name_input = input("Enter a movie name to visualize sentiment rating: ")

if movie_name_input in df['movie_name'].values:
    movie_row = df[df['movie_name'] == movie_name_input].iloc[0]
    feedbacks = [movie_row['feedback1'], movie_row['feedback2'], movie_row['feedback3'], movie_row['feedback4']]

    # Predict sentiment and rating per feedback
    feedback_df = pd.DataFrame({'feedback': feedbacks})
    feedback_df['cleaned'] = feedback_df['feedback'].apply(lambda x: preprocess(x) if isinstance(x, str) else "")
    X_fb = tfidf.transform(feedback_df['cleaned'])
    feedback_df['sentiment'] = model.predict(X_fb)
    feedback_df['sentiment'] = feedback_df['sentiment'].apply(lambda x: 'Positive' if x == 1 else 'Negative')
    feedback_df['rating'] = (model.predict_proba(X_fb)[:, 1] * 9 + 1).round(1)

    pos_count = (feedback_df['sentiment'] == 'Positive').sum()
    neg_count = (feedback_df['sentiment'] == 'Negative').sum()
    total = len(feedback_df)

    print(f"\nMovie: {movie_name_input}")
    print(feedback_df[['feedback', 'sentiment', 'rating']])
    print(f"\nPositive Reviews: {pos_count}")
    print(f"Negative Reviews: {neg_count}")
    print(f"Average Rating: {feedback_df['rating'].mean():.1f}/10")

    # --- Pie Chart by predicted sentiment ---
    plt.figure(figsize=(5,5))
    plt.pie(
        [pos_count, neg_count],
        labels=['Positive', 'Negative'],
        autopct=lambda p: f'{p:.1f}% ({int(p*total/100)})',
        colors=['lightgreen', 'salmon'],
        startangle=90
    )
    plt.title(f"Sentiment Breakdown for '{movie_name_input}'")
    plt.show()

else:
    print(f"\nMovie '{movie_name_input}' not found in the dataset.")
