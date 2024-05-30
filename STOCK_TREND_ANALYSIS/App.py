import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
import yfinance as yf
from yahoo_fin import stock_info
import pyttsx3
import speech_recognition as sr
from nsetools import Nse
import spacy
##Creating a dictionary of stock symbols with company names
ind_nifty500 = pd.read_csv(r"ind_nifty500list.csv")
symbol_dict= {}
a=0
for i in ind_nifty500["Company Name"]:
    
    
    symbol_dict[i] = ind_nifty500[ind_nifty500["Company Name"]==i]["Symbol"][a]+".NS"
    a=a+1


# Load spaCy English model
nlp = spacy.load("en_core_web_sm")
def Say(Text):
    engine = pyttsx3.init("sapi5")
    voices = engine.getProperty('voices')
    
    # Adjust voice selection based on available voices
    engine.setProperty('voice', voices[0].id)  # You can change [0] to select a different voice
    engine.setProperty('rate', 170)  # Adjust the speech rate as needed
    
    # Print the spoken text
    print("A.I:", Text)
    
    # Speak the text
    engine.say(Text)
    engine.runAndWait()
def Listen():
    r = sr.Recognizer()

    with sr.Microphone() as source:
        print("Listening......")
        r.pause_threshold = 1
        audio = r.listen(source,0,10)

    try:
        print("Recognizing.....")
        query = r.recognize_google(audio,language="en-ie")
        print(f"You Said : {query}")

    except:
        return ""
    
    query = str(query)
    return query


start = '2010-01-01'
end = '2023-11-20'

st.title('Stock Trend Prediction')
#fetching Organization name from the query
def get_stock_data(query):
     # Process the user query using spaCy
    doc = nlp(query)

    # Extract relevant information from the query
    organization_name = None

    # Look for the first entity with label "ORG" in the spaCy processed query
    for ent in doc.ents:
        if ent.label_ == "ORG":
            organization_name = ent.text
            break
    if organization_name:
        stock_symbol = get_stock_symbol(organization_name)
        if stock_symbol:
            return stock_symbol
             
        else:
            print(f"No stock symbol found for {organization_name}.")
            return None
    else:
        print("No organization name found in the query.")
        return None

def get_stock_symbol(organization_name):
    Symbol_lis = {}
    for i in symbol_dict:
        if organization_name in i:
            Symbol_lis[i] = symbol_dict[i]
    return Symbol_lis
   

    


def get_user_input():
    input_type = st.radio("Select Input Type:", ["Text Input", "Voice Input","Ticker"])
    
    if input_type == "Text Input":
        user_input = st.text_input('Enter Query')
        user_input = get_stock_data(user_input)
    if input_type == "Voice Input":
                 user_input = get_voice_input()
    if input_type == "Ticker":
        input = st.text_input('Enter Stock Ticker')
        user_input = {}
        for i in symbol_dict:
            if input in symbol_dict[i]:
                user_input[i] = symbol_dict[i]
    else :
        print("No Input Given")
    
    return user_input


def get_stock_summary(ticker):
    try:
        summary_data = stock_info.get_quote_table(ticker)
        summary_paragraph = "\n".join([f"{key}: {value}" for key, value in summary_data.items()])
        return summary_paragraph
    except Exception as e:
        return f"Error: {e}"
def get_stock_table(ticker):
    try:
        summary_data = stock_info.get_quote_table(ticker)
        return summary_data
    except Exception as e:
        return f"Error: {e}"

def get_voice_input():
    
    try:
        user_input = Listen()
        user_input = get_stock_data(user_input)
        return user_input
        
    except sr.UnknownValueError:
        st.warning("Sorry, I could not understand your speech. Please try again.")
        return None
        
    except sr.RequestError as e:
        st.error(f"Error with the speech recognition service: {e}")
        return None
        

# Get user input
user_input = get_user_input()
if user_input is not None:
 for i in user_input:
    st.subheader("COMPANY NAME",'r')
    st.subheader(i)
    st.subheader("TICKER",'g')
    st.subheader(user_input[i])



    try:
        df = yf.download(user_input[i], start=start, end=end)
        data_fetched = True
    except Exception as e:
        st.warning(f"Failed to fetch data from Yahoo Finance. Trying NSE data. Error: {e}")
        data_fetched = False





    ticker_symbol = user_input[i]
    stock_summary = get_stock_summary(ticker_symbol)
    stock_table = get_stock_table(ticker_symbol)
    st.write(stock_table)
    Say(stock_summary)


   #Describing Data

    st.subheader('Data from 2010-2023')
    st.write(df.describe())


#Visualizations

    st.subheader('Closing Price vs Time Chart')

    fig = plt.figure(figsize= (12,6))
    plt.plot(df.Close)
    st.pyplot(fig)


    st.subheader('Closing Price vs Time Chart with 100MA')
    ma100 = df.Close.rolling(100).mean()
    fig = plt.figure(figsize= (12,6))
    plt.plot(ma100)
    plt.plot(df.Close)
    st.pyplot(fig)


    st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
    ma100 = df.Close.rolling(100).mean()
    ma200 = df.Close.rolling(200).mean()
    fig = plt.figure(figsize= (12,6))
    plt.plot(ma100,'r')
    plt.plot(ma200,'g')
    plt.plot(df.Close,'b')
    st.pyplot(fig)

#splitting data into training and testing
    data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])


    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1))
    data_training_array= scaler.fit_transform(data_training)


# Loading the Model
    model = load_model('keras_model.h5')

#Testing Part

    past_100_days = data_training.tail(100)
    final_df = past_100_days.append(data_testing,ignore_index = True)
    input_data = scaler.fit_transform(final_df)

    x_test =[]
    y_test =[]

    for i in range(100,input_data.shape[0]):
         x_test.append(input_data[i-100:i])
         y_test.append(input_data[i,0])
    
    x_test , y_test = np.array(x_test),np.array(y_test)

    y_predicted = model.predict(x_test)

    scaler = scaler.scale_

    scale_factor = 1 /scaler[0]
    y_predicted = y_predicted * scale_factor
    y_test = y_test * scale_factor 

#Final Graph
    st.subheader('Predictions vs Original')
    fig2 = plt.figure(figsize=(12,6))
    plt.plot(y_test,'b',label = 'Original Price')
    plt.plot(y_predicted,'r',label = 'Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig2)

# Assuming y_predicted and y_test are arrays of predicted and actual prices

# Identify uptrends and downtrends
    trend_threshold = 0.02  # Adjust this threshold based on your data and model performance

    uptrends = y_predicted > (1 + trend_threshold) * y_test
    downtrends = y_predicted < (1 - trend_threshold) * y_test
# Count the occurrences of uptrends and downtrends
    uptrend_count = np.count_nonzero(uptrends)
    downtrend_count = np.count_nonzero(downtrends)

    total_samples = len(uptrends)

# Count the occurrences of uptrends and downtrends
    uptrend_count = np.count_nonzero(uptrends)
    downtrend_count = np.count_nonzero(downtrends)

    total_samples = len(uptrends)

# Calculate the percentage of uptrends and downtrends
    uptrend_percentage = (uptrend_count / total_samples) * 100
    downtrend_percentage = (downtrend_count / total_samples) * 100

# Convert percentages to strings
    uptrend_percentage_str = "{:.2f}%".format(uptrend_percentage)
    downtrend_percentage_str = "{:.2f}%".format(downtrend_percentage)

# Generate summary paragraphs
    uptrend_summary = f"The model predicts an uptrend in {uptrend_percentage_str} of the cases."
    downtrend_summary = f"The model predicts a downtrend in {downtrend_percentage_str} of the cases."

# Display summaries
    st.subheader('Trend Summary')
    st.write(uptrend_summary)
    st.write(downtrend_summary)
    Say(uptrend_summary)
    Say(downtrend_summary)
else:
    
    st.warning("No valid input found. Please try again.")