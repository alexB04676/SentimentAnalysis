import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
import os
import json
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
from joblib import dump

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


class TqdmSearchCV(RandomizedSearchCV):
    def fit(self, X, y, **fit_params):
        with tqdm(total=self.n_iter, desc="Randomized Search Progress") as pbar:
            for _ in super().fit(X, y, **fit_params).cv_results_['params']:
                pbar.update(1)
        return self

# Load the Dataset with relative paths and error handling
try:
    data_path = os.path.join(os.getcwd(), "cleaned_dataset.jsonl")
    df = pd.read_json(data_path, lines=True)
    text = df["text"]
    rate = df["rating"]
    if df.empty:
        raise ValueError("Dataset is empty. Maybe there was a problem in your preprocessing? ")
except FileNotFoundError:
    print("Error: cleaned_dataset.jsonl not found. Run Dataset_cleaner.py first.")
    exit()
except ValueError as e:
    print(e)
    exit()
    

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

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    tfidf_matrix, rate, test_size=0.2, random_state=42
)

# Define hyperparameter search space for Randomized Search
param_distributions = {
    'n_estimators': [50, 100, 200, 300, 500],  # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],           # Maximum depth of each tree
    'min_samples_split': [2, 5, 10],           # Minimum samples required to split a node
    'min_samples_leaf': [1, 2, 4],             # Minimum samples required at a leaf node
    'max_features': ['sqrt', 'log2']           # Number of features to consider at each split
}

# Initialize Random Forest model  
rf = RandomForestClassifier(random_state=42)


# Set up Randomized Search with Tqdm progress bar
random_search = TqdmSearchCV(
    estimator=rf,
    param_distributions=param_distributions,
    n_iter=50,               # Number of random hyperparameter combinations to test
    scoring='neg_mean_squared_error',  # Optimize for lowest mean squared error
    cv=5,                    # 5-fold cross-validation for reliable evaluation
    verbose=0,               # Suppress output clutter
    random_state=42,         # Ensure reproducibility
    n_jobs=-1                # Use all available CPU cores for faster computation
)

# Fit the model with the randomized search
random_search.fit(X_train, y_train)

print(f"Best hyperparameter Config{random_search.best_params_}")

# Retrieve the best model from Randomized Search
best_model = random_search.best_estimator_

# Predict on the test data using the best model
y_pred = best_model.predict(X_test)


# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

results = {
    "best_params": random_search.best_params_,
    "accuracy": accuracy,
    "mse": mse,
    "r2_score": r2
}
with open("results.json", "w") as f:
    json.dump(results, f, indent=4)
    print("Results saved in 'results.json'")


print("\nModel Performance with Best Parameters:")
print(f"Mean Squared Error (MSE): {mse:.3f}")
print(f"Accuracy: {accuracy:.3f}")

dump(best_model, "RandomForestSA.joblib")
print("Model Saved as 'RandomForestSA.joblib'")


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