import streamlit as st

st.title("Disaster-related tweets classifier")
from string import punctuation
import  pickle
import re
import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt_tab')
import emoji
from nltk.stem import PorterStemmer
ps=PorterStemmer()

def transform_text(text):
    text = emoji.demojize(text)
    text = text.lower()

    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


TV = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

input_tweet = st.text_area("Enter the tweet")


if st.button('Predict'):

    transformed_tweet = transform_text(input_tweet)
    vector_input = TV.transform([transformed_tweet])

    result = model.predict(vector_input)[0]

    if result == 1:
        st.header("The tweet is related to the Disasters")
        st.image("C:\\Users\\ay284\\Downloads\\vildisast.jpg")
    else:
        st.header("The tweet is not related to the disasters")
        st.image("C:\\Users\\ay284\\Downloads\\pexels-willsantos-2026960.jpg")





