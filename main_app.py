import numpy as np
import pandas as pd
import re
import string
from nltk.corpus import stopwords
import tensorflow as tf
from bert import tokenization
import joblib
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import load_model
import tensorflow_addons as tfa
import streamlit as st

#model = load_model('C:\Users\Onkar\Self_Case_Study_2-Deploy\lstm_model.h5')
model = load_model('C:\\Users\\Onkar\\Self_Case_Study_2-Deploy\\lstm_model.h5', custom_objects={'f1_score':tfa.metrics.F1Score(num_classes =  2, average = 'micro')})

class_names = [0, 1]

st.title('Prediction of Disaster Tweets')
st.markdown('Type a Tweet')

text = st.text_area('type a tweet here..')
submit = st.button('Predict')

if submit:

    if text is not None:

        #Text Preprocessing
        text = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", text)
        tags = re.compile('<.*?>') 
        text = re.sub(tags, '', text)
        emoji_pattern = re.compile("["u"\U0001F600-\U0001F64F"  u"\U0001F300-\U0001F5FF"  u"\U0001F680-\U0001F6FF"  u"\U0001F1E0-\U0001F1FF"  u"\U00002702-\U000027B0"u"\U000024C2-\U0001F251""]+", flags = re.UNICODE)
        text = re.sub(emoji_pattern, '', text)
        table = str.maketrans('', '', string.punctuation)
        text = text.translate(table)
        pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
        text = pattern.sub('', text)
        text = [text]

        max_sequence_length = 50
        max_words = 10000
        tokenizer = joblib.load('tokenizer.pkl')
        sequences = tokenizer.texts_to_sequences(text)
        padded = pad_sequences(sequences, padding = 'post', truncating = 'post', maxlen = max_sequence_length)

        pred = model.predict(padded)
        if pred[0][0] > pred[0][1]:
            final_pred = 0
            st.title('This tweet is not a disaster.')
        else:
            final_pred = 1
            st.title('This tweet is a real disaster.')
