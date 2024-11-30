import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
import nltk

# Download necessary NLTK data
nltk.download("stopwords", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("wordnet", quiet=True)

# Initialize tools
stop = set(stopwords.words("english"))  # Load English stopwords
lemmatizer = WordNetLemmatizer()  # Initialize the WordNet lemmatizer

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


# Generate POS tags for tokens and Lemmatize each word with its POS tag
def lemmatize_text(tokens):
    tagged_tokens = pos_tag(tokens)
    return [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in tagged_tokens]


# Read and preprocess the dataset
df = pd.read_json("C:/Users/ali/Projects/BasicSentimentAnalysis/Digital_Music.jsonl", lines=True)


# Retain only the "rating" and "text" columns for analysis
df = df.filter(["rating", "text"])

# Enable progress bars for Pandas operations
tqdm.pandas()


# Convert text to lowercase to standardize the format for processing
df["text"] = df["text"].progress_apply(lambda x: x.lower() if isinstance(x, str) else x)

# Remove special characters to clean the text for tokenization
df["text"] = df["text"].progress_apply(lambda x: re.sub(r"\W", " ", x) if isinstance(x, str) else x)

# Tokenize the "text" column into individual words for further analysis
df["text"] = df["text"].progress_apply(lambda x: word_tokenize(x) if isinstance(x, str) else x)

# Remove stopwords from the tokenized text to focus on meaningful words
df["text"] = df["text"].progress_apply(lambda tokens: [word for word in tokens if word not in stop])

# Lemmatize the tokens to normalize words to their base forms
df["text"] = df["text"].progress_apply(lemmatize_text)

# Write the cleaned dataset to a new JSONL file for future use
df.to_json('cleaned_dataset.jsonl', orient='records', lines=True)

# Print a confirmation message
print("Preprocessing complete!")
