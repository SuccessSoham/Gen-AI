import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Streamlit app configuration
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Gradient background style
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to right, #ff7e5f, #feb47b);
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App title and description
st.title("📈 Stock Price Predictor")
st.write("Upload your historical stock data (CSV or XLSX) to predict the next 7 days of stock prices using an LSTM model.")

# File uploader
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Read the uploaded file
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file, engine='openpyxl')

    # Display the uploaded data
    st.subheader("Uploaded Data")
    st.write(data.head())

    # Prepare the data
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date')
    data.set_index('Date', inplace=True)

    # Use only 'Close' price for LSTM
    close_data = data[['Close']].values

    # Normalize data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(close_data)

    # Create sequences
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        return np.array(X), np.array(y)

    seq_length = 60
    X, y = create_sequences(scaled_data, seq_length)

    # Split into train and test
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build LSTM model
    model = Sequential([
        LSTM(50, return_sequences=False, input_shape=(seq_length, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the model
    with st.spinner('Training the model...'):
        model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)
    st.success('Model training completed!')

    # Predict the next 7 days
    def predict_next_7_days(model, data, scaler, seq_length):
        predictions = []
        last_sequence = data[-seq_length:]
        for _ in range(7):
            input_data = np.array(last_sequence).reshape((1, seq_length, 1))
            input_scaled = scaler.transform(input_data.reshape(-1, 1)).reshape((1, seq_length, 1))
            prediction_scaled = model.predict(input_scaled)
            prediction = scaler.inverse_transform(prediction_scaled)[0][0]
            predictions.append(prediction)
            last_sequence = np.append(last_sequence[1:], prediction)
        return predictions

    next_7_days = predict_next_7_days(model, scaled_data, scaler, seq_length)

    # Display the predictions
    st.subheader("Predicted Stock Prices for the Next 7 Days")
    st.write(next_7_days)

    # Plot the predictions
    st.subheader("Stock Price Predictions")
    fig, ax = plt.subplots()
    ax.plot(data.index[-60:], close_data[-60:], label='Historical Prices')
    future_dates = pd.date_range(start=data.index[-1], periods=8, closed='right')
    ax.plot(future_dates, next_7_days, label='Predicted Prices', linestyle='--')
    ax.legend()
    st.pyplot(fig)