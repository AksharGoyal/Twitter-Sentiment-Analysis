import pandas as pd
import numpy as np
import sklearn
import streamlit as st
import pickle
# import matplotlib
# from IPython import get_ipython

# load the encoder and model object
with open('sentiment_analysis_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('text_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title="Twitter Sentiment Analysis", layout="wide")
st.markdown("<h1 style='text-align: center;'>Twitter Sentiment Analysis App</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: right;'>Built by Akshar Goyal with Streamlit</h3>", unsafe_allow_html=True)
st.markdown("""<p style='font-size: 1.25em'>The model is trained on twitter samples from the <code>nltk</code> library. It is a collection of tweets categorized either as positive or negative.<br>
            <strong>TF-IDF</strong> stands for Term Frequency - Inverse Document Frequency. It is a statistical measure that evaluates how important a word is to a document in a collection or corpus.<br>We use it to vectorize the tweets as input and then use our trained model to classify the sentiment of the person behind the tweet.
            </p>""", unsafe_allow_html=True)
def main():
    a, b = st.columns([0.5, 0.5])
    with a:
        tweet = st.text_input("Enter a tweet", "What an amazing app!")
    submit = st.button("Submit")
    if submit:
        transformed_text = vectorizer.transform([tweet])
        prediction = model.predict(transformed_text)[0]
        output = "Result: This tweet is " + prediction + ("😄" if prediction == 'positive' else "☹️")
        st.write(output)
    st.markdown("<h3>Code</h3>", unsafe_allow_html=True)
    st.markdown('You can view the code by clicking on the [link](https://github.com/AksharGoyal/Twitter-Sentiment-Analysis).', unsafe_allow_html=True)
    st.markdown('Link to the app: https://github.com/AksharGoyal/Twitter-Sentiment-Analysis', unsafe_allow_html=True)
    
if __name__ == '__main__':
  main()