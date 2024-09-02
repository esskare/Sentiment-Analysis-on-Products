# nltk.download()
#imports and first doing VADER MODEL
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re  # Import re for regex-based tokenization

plt.style.use('ggplot')

# # Define the custom tokenization function
# def simple_tokenize(text):
#     tokens = text.split()  # Basic tokenization by splitting on whitespace
#     return tokens

# Reading data 
df = pd.read_csv("Amazon_Unlocked_Mobile.csv")

# Show the specified index on the 'Reviews' row
print(df['Reviews'].values[0])  

# Print the shape of the dataset.
print(df.shape)

# Display the first few rows of the DataFrame
df = df.head(500)

# Quick EDA: Plot the count of reviews by stars
ax = df['Rating'].value_counts().sort_index().plot(
    kind='bar', 
    title='Count of Reviews by stars',
    figsize=(10, 5)
)
ax.set_xlabel('Review Stars')
plt.show()

##
# Basic NLTK Tokenization
example = df['Reviews'][10]
tokens = nltk.word_tokenize(example)  # Tokenize the 'example' variable
print(tokens)
tokens[:10] #first 10 words


tagged= nltk.pos_tag(tokens)
tagged[:10]
#putting them into entities

entities = nltk.chunk.ne_chunk(tagged)
entities.pprint()
#Vader Sentiment Scoring
#- Uses a "cluster of words" approach
#- Stop words are removed e.g. and, or
#- Each word is combined to a total score
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

sia = SentimentIntensityAnalyzer()
sia.polarity_scores('I am so happy!')
sia.polarity_scores(example)
df.head()
print(df.columns)
#Run polarity score on entire dataset
from tqdm.notebook import tqdm
import pandas as pd

# Ensure 'Reviews' column is string type and handle missing values
df['Reviews'] = df['Reviews'].astype(str)  # Convert all entries to string

# Dictionary for results
results = {}

for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['Reviews']
    name = row['Brand Name']
    
    if isinstance(text, str):  # Check if text is a string
        results[name] = sia.polarity_scores(text)
    else:
        results[name] = {"error": "Invalid text"}

#results
vaders = pd.DataFrame(results).T #.T flips the dataset horizontally
#sentiment analysis on the whole dataset of 500
vaders = vaders.merge(df, how='left', on='Brand Name')
vaders.head()
#ROBERTA MODEL
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
MODEL = f'cardiffnlp/twitter-roberta-base-sentiment'
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
#vader results on example
print(example)
# sia.polarity_scores(example)
