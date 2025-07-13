import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
import yfinance as yf
import datetime

# Title
st.title('📈 Stock Market Prediction using LSTM')

# Input stock ticker
ticker = st.text_input("Enter Stock Ticker (e.g. AAPL, GOOGL, TSLA):", value="AAPL")

# Load stock data
start = datetime.datetime(2012, 1, 1)
end = datetime.datetime.now()

if ticker:
    data = yf.download(ticker, start=start, end=end)
    st.subheader("Historical Data")
    st.write(data.tail())

    # Plotting
    st.subheader("Closing Price vs Time")
    fig = plt.figure(figsize=(10, 4))
    plt.plot(data.Close)
    st.pyplot(fig)

    st.info("Note: LSTM model needs to be pre-trained and loaded. You can integrate your trained model here.")
