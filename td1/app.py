import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download the VADER sentiment analysis model (only required once)
nltk.download("vader_lexicon")

# Create a Streamlit app
st.title("TD1 - Machine Learning for NLP")

# Input text box for user to enter text
user_input = st.text_area("Enter a sentence for sentiment analysis:", "")

# Initialize the sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to analyze sentiment and return the sentiment label
def analyze_sentiment(text):
    sentiment_scores = analyzer.polarity_scores(text)
    sentiment = sentiment_scores["compound"]
    if sentiment >= 0.05:
        return "Positive"
    elif sentiment <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# Perform sentiment analysis and display the result
if st.button("Analyze Sentiment"):
    if user_input:
        sentiment = analyze_sentiment(user_input)
        st.write(f"Sentiment: {sentiment}")
    else:
        st.warning("Enter text:")
