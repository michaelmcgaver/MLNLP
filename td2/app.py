import streamlit as st
import nltk
from nltk.tag import UnigramTagger
from nltk.corpus import treebank
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer

# Downloading necessary NLTK datasets
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

# Title and introduction
st.title("NLP Analysis with NLTK")
st.write("This application performs part-of-speech tagging and sentiment analysis on the text you enter.")

# User input
user_input = st.text_area("Enter your text here", "Today I feel so lucky and happy!")

# POS Tagging
if st.button('Analyze Text'):
    token = nltk.word_tokenize(user_input)
    tagged = nltk.pos_tag(token)
    st.write("Part-of-Speech Tagging:")
    st.write(tagged)

    # Here you can add more NLP analysis features like sentiment analysis

# Run this app with: streamlit run your_script_name.py
