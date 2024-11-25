import pandas as pd
import re
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords

stop = set(stopwords.words("english"))


# Read the JSONL file and retain only the relevant "rating" and "text" columns
# This ensures we work with cleaner data for processing
df = pd.read_json("C:/Users/ali/Projects/BasicSentimentAnalysis/Digital_Music.jsonl", lines = True)
df = df.filter(["rating", "text"])

# Convert text to lowercase to standardize the format for processing
df = df.apply(lambda col: col.map(lambda x: x.lower() if isinstance(x, str) else x))

# Remove special characters to clean the text for tokenization
df["text"] = df["text"].apply(lambda x: re.sub(r"\W", " ", x) if isinstance(x, str) else x)

# Tokenize the "text" column into individual words for further analysis
df["text"] = df["text"].apply(lambda x: word_tokenize(x) if isinstance(x, str) else x)

# Remove stopwords from the tokenized text to focus on meaningful words
df["text"] = df["text"].apply(lambda tokens: [word for word in tokens if word not in stop])

# Write the cleaned dataset to a new JSONL file for future use
df.to_json('cleaned_dataset.jsonl', orient='records', lines=True)

# print a confirmation message
print("Data Cleaned Succesfully!")