from sklearn.linear_model import Ridge
import nltk
import pandas as pd


data = pd.read_csv("ImdbReviewsAveragePS.csv")
print(data.shape)
print(data.head(1000))

