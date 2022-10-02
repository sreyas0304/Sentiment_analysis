import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as data
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
from keras.models import load_model
from PIL import Image
import tweepy as tw
import json
import re
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob

#Creating UI
start = st.sidebar.text_input('Start Date', '2010-01-01')
end = st.sidebar.text_input('End Date', '2021-09-05')

st.title('Stock Price Predictor')

image = Image.open('BEARBULL-removebg-preview.png')
st.image(image, use_column_width=True)

st.sidebar.header('User Input')
user_input = st.sidebar.selectbox('Enter Stock Ticker', ('ADANIPORTS.NS', 'TCS.NS', 'TATAMOTORS.NS',
                                                         'ASIANPAINT.NS', 'BAJAJ-AUTO.NS', 'LT.NS', 'BRITANNIA.NS',
                                                         'CIPLA.NS', 'COALINDIA.NS', 'DRREDDY.NS', 'GAIL.NS',
                                                         'ICICIBANK.NS', 'INFY.NS', 'ITC.NS'))

df = data.DataReader(user_input, 'yahoo', start, end)

st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Data from '+start+' - Latest')
st.write(df.describe())


#Preparing Training and Testing Datasets
data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):int(len(df))])

scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)

# Loading the model
model = load_model("C:/Users/Ansh Podar/OneDrive/Desktop/Models/keras_model_final.h5")

# Testing Part
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)

X_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    X_test.append(input_data[i - 100:i])
    y_test.append(input_data[i, 0])

X_test, y_test = np.array(X_test), np.array(y_test)

#Training Part
X_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):
    X_train.append(data_training_array[i - 100:i])
    y_train.append(data_training_array[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

# Sentiment Analysis
consumer_key = '7XKbRHJDMQoCq8OgP5cYVnlzi'
consumer_secret = 'o70yc5IR8Ugw0hXbUbc32tUBqEA9No5iJBjZGq1YSK8w4ggF68'
access_token = '1436920975786852353-iQwHMvu50Yj5Sb2EctLPNhKJ8fR9hU'
access_token_secret = 'jcNmJg9UCIppRwJBeG9DMSAJTIidgxJ6SAAmFv8F5Ke7o'

auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)

search_words_list = ["#inflation", '#indianpolitics', '#climatechange ', '#pharma','#interestrates', '#ModiGovt',
                     '@ShareMarketIndi', '#covidindia',
                     '@IncomeTaxIndia', '#indianstocks', '#BSE', '#demand']

final_tweets_list = []
tweets_data = []
for search_words in search_words_list:
    tweets = tw.Cursor(api.search_tweets,
                       q=search_words,
                       lang="en").items(10)
    l = []
    for tweet in tweets:
        l.append(tweet)
    columns = set()
    allowed_types = [str, int]
    # tweets_data = []
    for status in l:
        # print(status.text)
        # print(vars(status))
        status_dict = dict(vars(status))
        keys = status_dict.keys()
        single_tweet_data = {}
        for k in keys:
            try:
                v_type = type(status_dict[k])
            except:
                v_type = None
            if v_type != None:
                if v_type in allowed_types:
                    single_tweet_data[k] = status_dict[k]
                    columns.add(k)
        tweets_data.append(single_tweet_data)
        final_tweets_list.append(tweets_data)

    header_cols = list(columns)

df = pd.DataFrame(tweets_data, columns=header_cols)

sentiment = df[['text']]
sentiment = sentiment.iloc[:100]

def cleanText(text):
    text = text.lower()
    # Removes all mentions (@username) from the tweet since it is of no use to us
    text = re.sub(r'(@[A-Za-z0-9_]+)', '', text)

    # Removes any link in the text
    text = re.sub('http://\S+|https://\S+', '', text)

    # Only considers the part of the string with char between a to z or digits and whitespace characters
    # Basically removes punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Removes stop words that have no use in sentiment analysis
    text_tokens = word_tokenize(text)
    text = [word for word in text_tokens if not word in stopwords.words()]

    text = ' '.join(text)
    return text


sentiment['text'] = sentiment['text'].apply(cleanText)


def getSubjectivity(review):
    return TextBlob(review).sentiment.subjectivity
    # function to calculate polarity


def getPolarity(review):
    return TextBlob(review).sentiment.polarity


# function to analyze the reviews
def analysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'


sentiment['Polarity'] = sentiment['text'].apply(getPolarity)
sent_list = list(sentiment['Polarity'])

historic = X_train[1].tolist()


def flatten_list(_2d_list):
    flat_list = []
    # Iterate through the outer list
    for element in _2d_list:
        if type(element) is list:
            # If the element is of type list, iterate through the sublist
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list


historic = flatten_list(historic)

final_df = pd.DataFrame()

final_df['Historic'] = historic
final_df['Polarity'] = sent_list

input_array = np.array(final_df)


#Final Prediction
y_predicted = model.predict(X_test)

scaler = scaler.scale_
scale_factor = 1 / scaler[0]

y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor


#Visualizations
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
st.pyplot(fig2)
