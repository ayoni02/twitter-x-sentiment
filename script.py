# Importing Dependencies

import tweepy
import re
import pandas as pd
import nltk
nltk.download('stopwords')
import string
from nltk.corpus import stopwords
import joblib
import streamlit as st

stop_words = stopwords.words('english')

def authenticate( api_key, api_secrets, access_token, access_secret):
    auth = tweepy.OAuthHandler(api_key,api_secrets)
    auth.set_access_token(access_token,access_secret)
    api = tweepy.API(auth)
    try:
        api.verify_credentials()
        return 'Successful Authentication'
    except:
        return 'Failed authentication'
    
def go(query, api_key, api_secrets, access_token, access_secret):
    auth = tweepy.OAuth1UserHandler(api_key, api_secrets, access_token, access_secret)
    api = tweepy.API(auth)
    columns = ['User', 'Tweet', 'created_at']
    payd_data = []
    fetched_tweets = api.search_tweets(q=query, count=100, tweet_mode='extended', result_type='recent')
    for tweet in fetched_tweets:
        payd_data.append([tweet.user.screen_name, tweet.full_text, tweet.created_at])
    df = pd.DataFrame(payd_data, columns=columns)
    
    return df 

def preprocess(text, stem=False):
    # Remove link,user and special characters
    text = re.sub("@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+", ' ', str(text).lower()).strip()
    
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)

    return [word for word in nopunc.split() if word.lower() not in stop_words]

def model(df, path):
    pipeline = joblib.load(path)
    predict = pipeline.predict(df['Tweet'])
    df['Sentiment'] = predict

    return df

def get_auth():
    st.write('Please enter your twitter credentials')
    api_key = st.text_input('Enter your api key: ')
    api_secrets = st.text_input('Enter your api secret: ')
    access_token = st.text_input('Enter your access token: ')
    access_secret = st.text_input('Enter your access secret: ')
    return api_key, api_secrets, access_token, access_secret

def get_query():
    query = st.text_input('Enter the twitter handle without @: ', placeholder='e.g. ayoni02')
    return query

def main():
    st.title("check how your official twitter handle is doing")
    consumer_key, consumer_secret, access_token, access_token_secret = get_auth()
    twit_auth = authenticate(consumer_key, consumer_secret, access_token, access_token_secret)
    if twit_auth == 'Successful Authentication':
        st.write('Authentication Successful')
        query = get_query()
        
        if query:
            df = go(query, consumer_key, consumer_secret, access_token, access_token_secret)
            if len(df) > 0:
                
                pipeline = joblib.load('pipeline.pkl')
                df = model(df, pipeline)
                st.write(len(df))
                md = len(df[df['User'] != query])
                st.write(f'Number of tweets made by others on {query} is {md} out of {len(df)}')
                st.write('Others were made by {query}')
                
            else:
                st.write('No tweets found')
    else:
        st.write('Authentication Failed, please reload the page and rewrite your credentials')


main()


