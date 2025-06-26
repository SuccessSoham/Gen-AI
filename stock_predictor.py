import numpy as np
import pandas as pd
import openpyxl
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

# Streamlit UI setup
st.set_page_config(page_title="Stock Predictor", page_icon="📈", layout="wide")
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f0f0f, #1c1c1c, #2b2b2b);
        color: white;
    }
    h1, h2, h3, h4, h5, h6 { color: white; }
    </style>
""", unsafe_allow_html=True)

st.title("📈 Stock Price Predictor with Seasonality")
st.write("Forecast the next 7 closing prices using an LSTM model that incorporates seasonal features like weekday and month.")

# File upload
uploaded_file = st.file_uploader("Upload your historical stock data", type=["csv", "xlsx"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file, engine="openpyxl")
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df.sort_values('Date', inplace=True)
    df.set_index('Date', inplace=True)

    st.subheader("📋 Sample of Uploaded Data")
    st.write(df.tail())

    # Use last 180 data points for more recent trend
    df_recent = df[['Close']].copy().tail(180)

    # Add cyclical seasonality features
    df_recent['day_sin']   = np.sin(2 * np.pi * df_recent.index.dayofweek / 7)
    df_recent['day_cos']   = np.cos(2 * np.pi * df_recent.index.dayofweek / 7)
    df_recent['month_sin'] = np.sin(2 * np.pi * df_recent.index.month / 12)
    df_recent['month_cos'] = np.cos(2 * np.pi * df_recent.index.month / 12)

    features = ['Close', 'day_sin', 'day_cos', 'month_sin', 'month_cos']
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_recent[features])

    # Sequence prep
    def create_sequences(data, seq_len=60):
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i+seq_len])
            y.append(data[i+seq_len][0])  # Predict 'Close'
        return np.array(X), np.array(y)

    seq_len = 60
    X, y = create_sequences(scaled_data, seq_len)

    # Model
    model = Sequential([
        Input(shape=(seq_len, X.shape[2])),
        LSTM(64),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    with st.spinner("⏳ Training model..."):
        model.fit(X, y, epochs=50, batch_size=16, verbose=0, callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])
    st.success("🎯 Model trained successfully!")

    # Prediction
    def forecast_next_7(model, base_df, scaled_seq, scaler):
        future_preds = []
        input_seq = scaled_seq[-seq_len:].reshape(1, seq_len, -1)
        last_date = base_df.index[-1]

        for _ in range(7):
            pred_scaled = model.predict(input_seq, verbose=0)
            last_feat = input_seq[0, -1, 1:]
            combined = np.concatenate([pred_scaled, last_feat.reshape(1, -1)], axis=1)
            pred_actual = scaler.inverse_transform(combined)[0][0]
            future_preds.append(round(pred_actual, 2))

            # Move to next date
            next_date = last_date + pd.Timedelta(days=1)
            while next_date.weekday() >= 5:
                next_date += pd.Timedelta(days=1)

            seasonal = [
                np.sin(2 * np.pi * next_date.weekday() / 7),
                np.cos(2 * np.pi * next_date.weekday() / 7),
                np.sin(2 * np.pi * next_date.month / 12),
                np.cos(2 * np.pi * next_date.month / 12)
            ]
            new_row = np.concatenate([pred_scaled[0], seasonal])
            input_seq = np.append(input_seq[0][1:], [new_row], axis=0).reshape(1, seq_len, -1)
            last_date = next_date

        future_dates = pd.date_range(df_recent.index[-1] + pd.Timedelta(days=1), periods=7, freq='B')
        return future_preds, future_dates

    predictions, future_dates = forecast_next_7(model, df_recent, scaled_data, scaler)

    st.subheader("📊 Predicted Closing Prices (Next 7 Business Days)")
    for date, price in zip(future_dates, predictions):
        st.write(f"{date.strftime('%A, %d %b')} → ₹{price}")

    st.subheader("🧾 Forecast Plot")
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('#121212')
    ax.set_facecolor('#121212')

    ax.plot(future_dates, predictions, marker='o', linestyle='--', color='orange', label='Forecasted Close')
    ax.set_title("7-Day Stock Price Forecast with Seasonality", color='white')
    ax.set_xlabel("Date", color='white')
    ax.set_ylabel("Price ($)", color='white')
    ax.tick_params(axis='x', rotation=45, colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.legend()

    st.pyplot(fig)