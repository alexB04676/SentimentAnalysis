import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Plots the distribution of TF-IDF importance scores.
def plot_distribution(tfidf_sums, save_path = None):
    """
    Parameters:
    - tfidf_sums (array-like): Sum of TF-IDF scores across all documents for each feature.
    - save_path (str or None): If provided, saves the plot to this path. Otherwise, shows it.
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

""" # Calculate the 90th percentile threshold for TF-IDF sums and retain only the first 10%
threshold = np.percentile(tfidf_sums, 90)

# Create a mask to identify features whose importance exceeds the threshold
top_features_mask = tfidf_sums > threshold

# Filter the original TF-IDF matrix to retain only the top 10% most important features
filtered_tfidf = tfidf_matrix[: , top_features_mask]

# Display the dimensionality before and after filtering
print(f"Original number of features: {tfidf_matrix.shape[1]}")
print(f"Number of features after filtering: {filtered_tfidf.shape[1]}")"""



x_train, x_test, y_train, y_test = train_test_split(
    tfidf_matrix, rate, test_size= 0.2, random_state= 42
)


ridge_model = Ridge(alpha=1.0, random_state=42)
ridge_model.fit(x_train, y_train)

# Predict on test data
y_pred = ridge_model.predict(x_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

baseline_pred = [y_train.mean()] * len(y_test)
baseline_mse = mean_squared_error(y_test, baseline_pred)
baseline_r2 = r2_score(y_test, y_pred)

print(f"Baseline MSE: {baseline_mse}")
print(f"baseline R2 Score: {baseline_r2}")
print("Model Performance:")
print(f"Mean Squared Error (MSE): {mse:.3f}")
print(f"R² Score: {r2:.3f}")

x_train_dense = x_train.toarray()

# Calculate the sum of TF-IDF scores per document
sum_tfidf_per_doc = np.sum(x_train_dense, axis=1)

# Scatterplot: Sum TF-IDF score vs Ratings
plt.scatter(sum_tfidf_per_doc, y_train, alpha=0.5)
plt.xlabel("Sum of TF-IDF Scores (per document)")
plt.ylabel("Rating")
plt.title("Sum TF-IDF vs Rating")
plt.show()
