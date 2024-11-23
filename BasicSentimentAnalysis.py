from sklearn.linear_model import Ridge
import nltk
import pandas as pd
import json


df = pd.read_json("C:/Users/ali/Projects/BasicSentimentAnalysis/Digital_Music.jsonl", lines = True)
print(df.shape)
print(df.head(100))
