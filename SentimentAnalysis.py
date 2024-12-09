import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import matplotlib

# Load the dataset
df = pd.read_json("C:/Users/ali/Projects/SentimentAnalysis/cleaned_dataset.jsonl", lines = True)
text = df["text"]
rate = df["rating"]


# Initialize TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),  # Includes both unigrams and bigrams
    stop_words='english' # removes common english stopwords (we already did that in data cleaning, but just in case...)
)


# Transform the text data into a sparse TF-IDF matrix
X = vectorizer.fit_transform(text)

# Extract feature names (words/bigrams) and compute their aggregate importance
feature_names = vectorizer.get_feature_names_out()
tfidf_sums = X.sum(axis=0).A1 # Aggregate importance (sum of TF-IDF scores) for each feature


# Plot the distribution of feature importance
matplotlib.use('Agg')
plt.hist(tfidf_sums, bins=50)
plt.title("TF-IDF Importance Distribution")
plt.xlabel("Aggregate Importance (Sum of TF-IDF)")
plt.ylabel("Frequency")
plt.savefig('tfidf_distribution.png')