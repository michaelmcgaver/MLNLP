import streamlit as st
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn

# Definire una funzione che mappa i tag del part-of-speech di treebank ai tag di WordNet
def treebank_to_wn(tag):
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('V'):
        return wn.VERB
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    else: 
        return None

def eval_sentiment(review):
    sentiment = 0.0
    tokens_count = 0
    lemmatizer = WordNetLemmatizer()
    token = nltk.word_tokenize(review)
    after_tagging = nltk.pos_tag(token)
    for word, tag in after_tagging:
        wn_tag = treebank_to_wn(tag)
        if wn_tag not in (wn.ADJ, wn.ADV):
            continue
        lemma = lemmatizer.lemmatize(word, pos=wn_tag)
        if not lemma:
            continue
        synsets = wn.synsets(lemma, pos=wn_tag)
        if not synsets:
            continue
        swn_synset = swn.senti_synset(synsets[0].name())
        sentiment += swn_synset.pos_score() - swn_synset.neg_score()
        tokens_count += 1

    return sentiment

st.title("Sentiment Analysis App")
st.write("Enter the text you'd like to analyze for sentiment.")

user_input = st.text_area("Text to analyze", "Type Here...")

if st.button('Evaluate Sentiment'):
    if user_input:
        result = eval_sentiment(user_input)
        st.write("The sentiment score of the input is:", result)
    else:
        st.error("ERROR")
