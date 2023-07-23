import streamlit as st
from textblob import TextBlob
import plotly.graph_objects as go
from wordcloud import WordCloud
import pandas as pd
import re
import time
from basc_py4chan import Board
from datetime import datetime, timedelta
import numpy as np
from PIL import Image


# Define a list of common English words
english_words = ["I", "THE", "AND", "OF", "TO", "IN", "THAT","FUCK", "IS", "IT", "FOR", "AS", "WAS", "WITH", "BE", "BY", "ON", "NOT", "HE", "THIS", "BUT", "ARE", "OR", "HIS", "AN", "THEY", "WHICH", "AT", "ALL", "FROM", "WE", "HAS", "NO", "WERE", "SO", "IF", "OUT", "UP", "A"]

def scrape_4chan():
    # Define the board you want to scrape
    board = Board('biz')

    # Get the current time
    now = datetime.now()

    # Define the time one month ago
    one_month_ago = now - timedelta(days=7)

    # Get the list of all threads on the board
    threads = board.get_all_threads()

    # Create an empty DataFrame
    data = pd.DataFrame(columns=['Thread ID', 'Thread Title', 'Thread Creation Time', 'Post ID', 'Post Content'])

    # Iterate through the threads
    for thread in threads:
        # Get the timestamp of the thread's creation time
        thread_time = thread.topic.datetime

        # Check if the thread was created within the last week
        if thread_time > one_month_ago:
            # Extract the desired information from the thread
            thread_id = thread.topic.post_id
            thread_title = thread.topic.subject
            thread_creation_time = thread_time.strftime("%d/%m/%Y")

            # Iterate through the posts in the thread
            for post in thread.all_posts:
                post_id = post.post_id
                post_content = post.text_comment

                # Append the data to the DataFrame
                new_row = pd.DataFrame({'Thread ID': [thread_id], 
                        'Thread Title': [thread_title], 
                        'Thread Creation Time': [thread_creation_time], 
                        'Post ID': [post_id], 
                        'Post Content': [post_content]})
                
                data = pd.concat([data,new_row],ignore_index=True)

                # If the DataFrame has reached 500 rows, break the loop
                if len(data) >= 500:
                    break

            # If the DataFrame has reached 500 rows, break the loop
            if len(data) >= 500:
                break

            # Add a delay to avoid overloading the server
            time.sleep(1)
    return data

def process_data(data):
    # Extract tickers from the 'Post Content'
    data['Tickers'] = data['Post Content'].apply(extract_tickers)

    # Analyze the sentiment of the 'Post Content'
    data['Sentiment'] = data['Post Content'].apply(get_sentiment)
    return data

def extract_tickers(text):
    """
    Extract tickers from the given text.
    """
    # Find all uppercase words with 2-4 characters
    potential_tickers = re.findall(r'\b[A-Z]{2,4}\b', text)
    
    # Filter out common English words
    tickers = [word for word in potential_tickers if word not in english_words]
    
    return tickers

def get_sentiment(text):
    """
    Analyze the sentiment of the given text.
    """
    blob = TextBlob(text)
    
    return blob.sentiment.polarity

def main():
    st.title("4chan biz Board Analysis")

    data = scrape_4chan()
    data = process_data(data)

    # Display the DataFrame
    st.dataframe(data)

    # 1. Ticker Counts
    tickers = data['Tickers'].explode().value_counts()

    fig = go.Figure([go.Bar(x=tickers.index[:15], y=tickers.values[:15])])
    fig.update_layout(title_text='Top 15 Most Mentioned Tickers', xaxis_title='Ticker', yaxis_title='Count')
    st.plotly_chart(fig)

    # 2. Sentiment Distribution
    fig = go.Figure(data=[go.Histogram(x=data['Sentiment'], nbinsx=20)])
    fig.update_layout(title_text='Sentiment Distribution', xaxis_title='Sentiment', yaxis_title='Count')
    st.plotly_chart(fig)

    # 3. Word Cloud
    text = ' '.join(post for post in data['Post Content'])
    wordcloud = WordCloud(background_color='white').generate(text)
    # Convert the word cloud to an image
    wordcloud_image = Image.fromarray(wordcloud.to_array())

    # Display the image
    st.image(wordcloud_image, caption='Word Cloud')

    # Create a dropdown select box for the tickers
    tickers = data['Tickers'].explode().unique()
    
    # Filter out nan values
    tickers = [ticker for ticker in tickers if str(ticker) != 'nan']

    selected_ticker = st.selectbox('Select a Ticker', options=tickers)

    # Filter the data for the selected ticker
    filtered_data = data[data['Tickers'].apply(lambda x: selected_ticker in x)]

    # Display the sentiment scores for the selected ticker
    st.header(f"Sentiment Scores for {selected_ticker}")
    st.dataframe(filtered_data['Sentiment'])

if __name__ == "__main__":
    main()
