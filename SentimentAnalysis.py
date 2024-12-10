import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import numpy as np

# Plots the distribution of TF-IDF importance scores.
def plot_distribution(tfidf_sums, save_path = None):
    """
    Parameters:
    - tfidf_sums (array-like): Sum of TF-IDF scores across all documents for each feature.
    
    Returns:
    None
    """
    plt.hist(tfidf_sums, bins=50)
    plt.title("TF-IDF Importance Distribution")
    plt.xlabel("Aggregate Importance (Sum of TF-IDF)")
    plt.ylabel("Frequency")

    if save_path:
        plt.savefig(save_path)
        print(f"Distribution saved as '{save_path}'.")
    else:
        plt.show()

# a function to ask user for either saving or just showing the TF-IDF distribution
def distribution_config(tfidf_sums):    
    distribution_options = input("""Type "Show" To Simply See The Importance Score Distribution Or "Save" To Save it as a Seperate File: """)
    if distribution_options.lower() == "save":
        plot_distribution(tfidf_sums, save_path="distribution.png")
        print("""Distribution Saved as "distribution.png".""")
    elif distribution_options.lower() == "show":
        plot_distribution(tfidf_sums, save_path=None)
    else:
        print("invalid input, Saving Distribution as a seperate file by default")
        plot_distribution(tfidf_sums, save_path="distribution.png")

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
tfidf_matrix = vectorizer.fit_transform(text)

# Extract feature names (words/bigrams) and compute their aggregate importance
feature_names = vectorizer.get_feature_names_out()
tfidf_sums = tfidf_matrix.sum(axis=0).A1 # Aggregate importance (sum of TF-IDF scores) for each feature

distribution_config(tfidf_sums)

# Calculate the 90th percentile threshold for TF-IDF sums and retain only the first 10%
threshold = np.percentile(tfidf_sums, 90)

# Create a mask to identify features whose importance exceeds the threshold
top_features_mask = tfidf_sums > threshold

# Filter the original TF-IDF matrix to retain only the top 10% most important features
filtered_tfidf = tfidf_matrix[: , top_features_mask]


# Display the dimensionality before and after filtering
print(f"Original number of features: {tfidf_matrix.shape[1]}")
print(f"Number of features after filtering: {filtered_tfidf.shape[1]}")