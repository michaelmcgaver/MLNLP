import streamlit as st
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon for sentiment analysis
nltk.download('vader_lexicon')

# Initialize the Sentiment Intensity Analyzer
sia = SentimentIntensityAnalyzer()

# Title and description of the app
st.title("Sentiment Analysis App")
st.write("This app uses the NLTK library to analyze sentiment of the text you enter.")

# User input for text analysis
user_input = st.text_area("Enter text here:", "Type your text...")

# When the 'Analyze' button is clicked, analyze the sentiment of the input text
if st.button('Analyze Sentiment'):
    if user_input:
        # Perform sentiment analysis
        sentiment_scores = sia.polarity_scores(user_input)
        # Display results
        st.subheader("Sentiment Analysis Results:")
        st.write(sentiment_scores)
    else:
        st.error("Error!")
