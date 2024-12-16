import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
import os

# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz

# Tree Visualisation
from sklearn.tree import export_graphviz

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
        print("invalid input, Showing you the score distributions by default")
        plot_distribution(tfidf_sums, save_path=None)

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

"distribution_config(tfidf_sums)"

# Calculate the 90th percentile threshold for TF-IDF sums and retain only the first 10%
threshold = np.percentile(tfidf_sums, 90)

# Create a mask to identify features whose importance exceeds the threshold
top_features_mask = tfidf_sums > threshold

# Filter the original TF-IDF matrix to retain only the top 10% most important features
filtered_tfidf = tfidf_matrix[: , top_features_mask]

filtered_feature_names = vectorizer.get_feature_names_out()[top_features_mask]

# Display the dimensionality before and after filtering
print(f"Original number of features: {tfidf_matrix.shape[1]}")
print(f"Number of features after filtering: {filtered_tfidf.shape[1]}")

X_train, X_test, y_train, y_test = train_test_split(
    filtered_tfidf, rate, test_size=0.2, random_state=42
)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

"""# Create a directory to save the trees
output_dir = "decision_trees"
os.makedirs(output_dir, exist_ok=True)

# Render and save the first 3 decision trees as PDFs
for i in range(3):
    tree = rf.estimators_[i]
    dot_data = export_graphviz(tree,
                               feature_names=filtered_feature_names,  # Filtered feature names
                               filled=True,  
                               max_depth=2, 
                               impurity=False, 
                               proportion=True)
    # Save graph to PDF
    graph = graphviz.Source(dot_data)
    output_path = os.path.join(output_dir, f"tree_{i+1}.pdf")
    graph.render(output_path, cleanup=True)  # Save and clean up temporary files
    print(f"Decision tree {i+1} saved to '{output_path}'.")"""
    
