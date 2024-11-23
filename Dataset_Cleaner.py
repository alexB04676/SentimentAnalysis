import pandas as pd
import json

# read the jsonl file and filter it out to only include "rating" and "text" columns for cleaner data
df = pd.read_json("C:/Users/ali/Projects/BasicSentimentAnalysis/Digital_Music.jsonl", lines = True)
df = df.filter(["rating", "text"])

# write it to a new jsonl file to retain the old dataset in case of need
df.to_json('cleaned_dataset.jsonl', orient='records', lines=True)

# print a confirmation message
print("Data Cleaned Succesfully!")