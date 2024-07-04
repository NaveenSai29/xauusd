import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st
import time
from tensorflow.keras.models import load_model
import joblib
import pytz
import requests
import csv
from datetime import datetime
st.set_page_config(
    page_title="Real-Time XAU-USD Dashboard",
    page_icon="✅",
    layout="wide",
)
st.title("Real-Time XAU-USD Price Candlestick Chart with KPIs")
st.markdown(
    """
    <style>
    body {
        background-color: #6495ED; /* Blue background */
    }
    div[data-testid="stTable"] {
        background-color: #90EE90 !important; /* Green KPI table background */
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <style>
    /* Style for News Table */
    div[data-testid="stHorizontalBlock"] div[data-testid="stTable"] {
        border: 1px solid black !important; /* Black border for news table */
        background-color: #ADD8E6  !important; /* Gray background for news table */
    }
    /* Style for Sentiment */
    .positive-sentiment {
        background-color: #90EE90 !important; /* Green background for positive sentiment */
        color: black !important; /* Set font color to black */
    }
    .negative-sentiment {
        background-color: #FF6347 !important; /* Red background for negative sentiment */
        color: black !important; /* Set font color to black */
    }
    </style>
    """,
    unsafe_allow_html=True
)


left_column, center_column,right_column = st.columns([5, 1,6])
#left_column.title("Real-Time XAU-USD KPIs and News")

kpi_table = left_column.empty()
news_table = left_column.empty()
action_table = center_column.empty()

#right_column.title("Real-Time XAU-USD Price Chart")
graph_placeholder = right_column.empty()
# Set the stock ticker symbol for Yahoo Finance
stock_symbol = "XAU-USD"


# Function to fetch live stock data
def create_sequences_for_1min(data, seq_length):
    sequences = []
    for i in range(seq_length, len(data)):
        sequences.append(data[i - seq_length:i, :])
    return np.array(sequences)


def predict_next_close(new_data_1):
    new_data = new_data_1[['Open', 'High', 'Low', 'Close']].pct_change()
    sequence_length = 30
    # Load the model
    model = load_model('next_close_prediction_model.h5')

    # Load the scaler object
    scaler = joblib.load('scaler.pkl')

    # Assume new_data is your new DataFrame with 'Open', 'High', 'Low', 'Close' columns
    new_values = new_data[['Open', 'High', 'Low', 'Close']].values
    new_values_1 = new_data_1[['Open', 'High', 'Low', 'Close']].values

    # Scale the new data
    scaled_new_data = scaler.transform(new_values)

    # Create sequences for the new data
    new_sequences = create_sequences_for_1min(scaled_new_data, sequence_length)

    # Extract the last sequence as the input for prediction
    new_input = new_sequences[-1, :-1].reshape(1, sequence_length - 1, 4)

    # Perform prediction
    predicted_scaled_change = model.predict(new_input)

    # Inverse transform the predicted values to get percentage change
    predicted_change = scaler.inverse_transform(predicted_scaled_change)[0][-1]

    # Fetch the last 'Close' price from your data
    last_close_price = new_values_1[-1][-1]

    # Apply the predicted percentage change to get the predicted next 'Close' price
    predicted_close = last_close_price * (1 + predicted_change)

    return predicted_close
    
    
def create_sequences_5min(data, seq_length):
    sequences = []
    for i in range(seq_length, len(data) -4):
        sequences.append(data[i - seq_length:i +4, :])
    return np.array(sequences)



def predict_next_close_5min(new_data_1):
    new_data = new_data_1[['Open', 'High', 'Low', 'Close']].pct_change()
    sequence_length = 30
    # Load the model
    model = load_model('next_close_prediction_model_5min.h5')

    # Load the scaler object
    scaler = joblib.load('scaler.pkl')

    # Assume new_data is your new DataFrame with 'Open', 'High', 'Low', 'Close' columns
    new_values = new_data[['Open', 'High', 'Low', 'Close']].values
    new_values_1 = new_data_1[['Open', 'High', 'Low', 'Close']].values

    # Scale the new data
    scaled_new_data = scaler.transform(new_values)

    # Create sequences for the new data
    input_sequences = create_sequences_5min(scaled_new_data, sequence_length)

    # Extract the last sequence as the input for prediction
    #new_input = new_sequences[-1, :-1].reshape(1, sequence_length - 1, 4)
    new_input = input_sequences.reshape(input_sequences.shape[0], input_sequences.shape[1], input_sequences.shape[2])

    print(new_input)
    # Perform prediction
    predicted_scaled_change = model.predict(new_input)
    print(predicted_scaled_change.shape,'predicted_scaled_change--------------------------------')

    # Inverse transform the predicted values to get percentage change
    predicted_change = scaler.inverse_transform(predicted_scaled_change)[-1][-1]

    # Fetch the last 'Close' price from your data
    last_close_price = new_values_1[-1][-1]

    # Apply the predicted percentage change to get the predicted next 'Close' price
    predicted_close_5min = last_close_price * (1 + predicted_change)

    return predicted_close_5min 
    
    
    
    
    
    


def give_minute_data():
    url = 'https://www.goldapi.io/api/XAU/USD'
    headers = {
        'x-access-token': 'goldapi-2zx4418lpl6rk8b-io'
    }

    csv_filename = 'gold_prices.csv'

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        price = data.get('price')
        timestamp = data.get('timestamp')

        # Convert epoch timestamp to a readable date/time format
        # readable_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        indian_tz = pytz.timezone('Asia/Kolkata')
        readable_time = datetime.utcnow().replace(tzinfo=pytz.utc).astimezone(indian_tz).strftime('%Y-%m-%d %H:%M:%S')

        with open(csv_filename, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([readable_time, price])

    df = pd.read_csv(csv_filename, names=['Timestamp', 'Price'])  # Read the CSV file
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])  # Convert 'Timestamp' to datetime

    # Filter data for today's date
    today_date = datetime.now().date()
    today_data = df[df['Timestamp'].dt.date == today_date]

    # Set 'Timestamp' as the DataFrame index
    today_data.set_index('Timestamp', inplace=True)

    # Resample data into minute candlesticks
    minute_candlesticks = today_data['Price'].resample('1Min').ohlc()

    # Create a new DataFrame including Timestamp and OHLC data
    stock_data_with_timestamp = pd.DataFrame({
        'Timestamp': minute_candlesticks.index,  # Include the Timestamp
        'Open': minute_candlesticks['open'],
        'High': minute_candlesticks['high'],
        'Low': minute_candlesticks['low'],
        'Close': minute_candlesticks['close']
    })

    return stock_data_with_timestamp


def get_xausd_news():
    url = "https://forexnewsapi.com/api/v1"
    params = {
        "currencypair": "XAU-USD",
        "items": 5,
        "token": "ap9uqgnowl9ogx6tnt2kqwomzhiahpy3n72o7xhs"
    }

    response = requests.get(url, params=params)
    print(response)

    if response.status_code == 200:
        data = response.json()
        data_dataframe =  pd.DataFrame(data['data'])
        df = data_dataframe[['date', 'title', 'sentiment']]
        filtered_df = df[df['sentiment'] != 'Neutral']
        return filtered_df

# Create an empty figure
fig = go.Figure()



# Loop to continuously update the candlestick chart with real-time stock data
last_time = 0
news = 'no_news'
sentiment = 'neutral'
action = 'NO TRADE'
time_trade = 'NO SIGNAL'
trade_data_list = [{'Time': 0, 'Action': 0}]
trade_data_list_news = [{'Time': 0, 'Action': 0}]
trade_data_list_lstm = [{'Time': 0, 'Action': 0}]
while True:
    stock_data = give_minute_data().dropna()
    if stock_data.index[-1] != last_time:
        last_time = stock_data.index[-1]
        news_df = get_xausd_news()

        pred_value = 0
        pred_value_5min = 0
        if len(stock_data) >= 30:
        
        #try:
            pred_value = predict_next_close(stock_data)
            pred_value_5min = predict_next_close_5min(stock_data)
            print(pred_value_5min,'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
        #except Exception as e:
            #print(e)


    trace = go.Candlestick(x=stock_data.index,
                           open=stock_data['Open'],
                           high=stock_data['High'],
                           low=stock_data['Low'],
                           close=stock_data['Close'],
                           name=stock_symbol)

    fig = go.Figure(data=[trace])

    # Update the layout of the figure
    fig.update_layout(title=f'{stock_symbol} Real-Time XAU-USD Price',
                      xaxis_title='Time',
                      yaxis_title='Price',
                      xaxis_rangeslider_visible=False)

    # Display the plot using Streamlit
    with graph_placeholder:
        st.plotly_chart(fig, use_container_width=True)

    kpi_df = pd.DataFrame({
            'Open': [str(stock_data['Open'].iloc[-1])],
            'High': [str(stock_data['High'].iloc[-1])],
            'Low': [str(stock_data['Low'].iloc[-1])],
            'Close': [str(stock_data['Close'].iloc[-1])],
            'Predicted Value': [str(round(pred_value, 2))],
            'Predicted Value_5min': [str(round(pred_value_5min, 2))]
        })


    def color_predicted_value(val):
        if float(val) < float(kpi_df['Close']):
            color = 'red'
        elif float(val) > float(kpi_df['Close']):
            color = 'green'
        else:
            color = 'black'
        return f'color: {color}'


    styled_kpi_df = kpi_df.style.applymap(color_predicted_value, subset=['Predicted Value','Predicted Value_5min'])

    with kpi_table:
        st.write("Real-Time XAU-USD KPIs and Predicted Value")
        st.table(styled_kpi_df)
    with action_table:
        if round(pred_value, 2) > stock_data['Close'].iloc[-1] and news_df['sentiment'].iloc[0] == 'Positive':
            action = 'BUY'
            indian_tz = pytz.timezone('Asia/Kolkata')
            time_trade = datetime.utcnow().replace(tzinfo=pytz.utc).astimezone(indian_tz).strftime('%Y-%m-%d %H:%M:%S')
            print(action)
            st.markdown(
                f'<p style="color:green;">{time_trade}   {action}</p>',
                unsafe_allow_html=True
            )
        elif round(pred_value, 2) < stock_data['Close'].iloc[-1] and news_df['sentiment'].iloc[0] == 'Negative':
            action = 'SELL'
            indian_tz = pytz.timezone('Asia/Kolkata')
            time_trade = datetime.utcnow().replace(tzinfo=pytz.utc).astimezone(indian_tz).strftime('%Y-%m-%d %H:%M:%S')
            print(action)
            st.markdown(
                f'<p style="color:red;">{time_trade}   {action}</p>',
                unsafe_allow_html=True
            )
        if trade_data_list[-1]['Action'] !=action:
            trade_data_list.append({'Time': time_trade, 'Action': action})

        trade_data = pd.DataFrame(trade_data_list)
        trade_data.to_csv('trade_data_combine.csv', index=False)


        #for news sentiment
        if news_df['sentiment'].iloc[0] == 'Positive':
            action_news = 'BUY'
            indian_tz = pytz.timezone('Asia/Kolkata')
            time_trade = datetime.utcnow().replace(tzinfo=pytz.utc).astimezone(indian_tz).strftime('%Y-%m-%d %H:%M:%S')

        elif news_df['sentiment'].iloc[0] == 'Negative':
            action_news = 'SELL'
            indian_tz = pytz.timezone('Asia/Kolkata')
            time_trade = datetime.utcnow().replace(tzinfo=pytz.utc).astimezone(indian_tz).strftime('%Y-%m-%d %H:%M:%S')

        if trade_data_list_news[-1]['Action'] !=action_news:
            trade_data_list_news.append({'Time': time_trade, 'Action': action_news})

        trade_data_news = pd.DataFrame(trade_data_list_news)
        trade_data_news.to_csv('trade_data_news.csv', index=False)



        # for lstm
        if round(pred_value, 2) > stock_data['Close'].iloc[-1] :
            action_lstm = 'BUY'
            indian_tz = pytz.timezone('Asia/Kolkata')
            time_trade = datetime.utcnow().replace(tzinfo=pytz.utc).astimezone(indian_tz).strftime('%Y-%m-%d %H:%M:%S')

        elif round(pred_value, 2) < stock_data['Close'].iloc[-1] :
            action_lstm = 'SELL'
            indian_tz = pytz.timezone('Asia/Kolkata')
            time_trade = datetime.utcnow().replace(tzinfo=pytz.utc).astimezone(indian_tz).strftime('%Y-%m-%d %H:%M:%S')

        if trade_data_list_lstm[-1]['Action'] !=action_lstm:
            trade_data_list_lstm.append({'Time': time_trade, 'Action': action_lstm})

        trade_data_lstm = pd.DataFrame(trade_data_list_lstm)
        trade_data_lstm.to_csv('trade_data_lstm.csv', index=False)


    with news_table:
        def color_sentiment(val):
            if val == 'Positive':
                color = 'green'
            elif val == 'Negative':
                color = 'red'
            else:
                color = 'black'
            return f'color: {color}'


        styled_news_df = news_df.style.applymap(color_sentiment, subset=['sentiment'])
        st.write("News and Sentiment")
        st.table(styled_news_df)
    time.sleep(1)