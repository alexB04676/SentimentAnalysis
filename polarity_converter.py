import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('punkt_tab')


# initialize VADER
analyzer = SentimentIntensityAnalyzer()

# function to tokenize sentences, give each word polarity scores and return an average for the entire sentence
def vader_wordpolarity(sentence):
    words = word_tokenize(sentence)
    if len(words) < 0:
        return 0
    scores = [analyzer.polarity_scores(word)['compound'] for word in words]
    return sum(scores) / len(scores)

# Read the dataset
df = pd.read_csv('C:/Users/ali/Projects/BasicSentimentAnalysis/ImdbDataset.csv')
print("Columns in the dataset:", df.columns)  # Debug: Check column names

# Check if 'review' column exists
if 'review' not in df.columns:
    raise ValueError("The 'review' column is not in the dataset.")

# Apply the function to each review
df['polarity_score'] = df['review'].apply(vader_wordpolarity)

# Verify if the column was added successfully
print("Updated columns:", df.columns)  # Debug: Check updated column names

# Save the updated dataset with both reviews and polarity scores
df[['review', 'polarity_score']].to_csv('updated_dataset_word_avg.csv', index=False)

print("New dataset with reviews and average polarity scores saved!")