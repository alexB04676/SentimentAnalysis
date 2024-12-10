import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import matplotlib

# Plots the distribution of TF-IDF importance scores.
def plot_distribution(tfidf_sums, save_path=None):
    """
    Parameters:
    - tfidf_sums (array-like): Sum of TF-IDF scores across all documents for each feature.
    - save_path (str): File path to save the plot. If None, the plot is just displayed.
    
    Returns:
    None
    """
    plt.hist(tfidf_sums, bins=50)
    plt.title("TF-IDF Importance Distribution")
    plt.xlabel("Aggregate Importance (Sum of TF-IDF)")
    plt.ylabel("Frequency")
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


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

distribution_options = input("""Type "Show" To Simply See The Importance Score Distribution Or "Save" To Save it as a Seperate File: """)
if distribution_options.lower() == "save":
    plot_distribution(tfidf_sums, save_path="distribution.png")
    print("""Distribution Saved as "distribution.png".""")
elif distribution_options.lower() == "show":
    plot_distribution(tfidf_sums, save_path=None)
else:
    print("invalid input, Saving Distribution as a seperate file by default")
    plot_distribution(tfidf_sums, save_path="distribution.png")