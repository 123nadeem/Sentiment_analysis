import streamlit as st
import pickle
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

ps = stopwords.words('english')

model = pickle.load(open('C:/Users/nadee/NLP/Sentiment_analysis/sentiment_analysis.pkl','rb'))

st.title('Sentiment Analysis Model')

review = st.text_input('Enter your Review')

def preprocess_review(review):
    review = BeautifulSoup(review, "html.parser").get_text()
    review = review.lower()
    review = ' '.join([word for word in review.split() if word not in ps])
    return review

processed_review = preprocess_review(review)

submit = st.button('Predict')

if submit:
    prediction = model.predict([processed_review])[0]
    st.write("Predicted Sentiment:", prediction)
