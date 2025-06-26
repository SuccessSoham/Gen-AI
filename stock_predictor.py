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

# Streamlit UI
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

st.title("📈 Stock Price Predictor (Holiday-Aware)")
st.write("Forecast the next 7 trading days using an LSTM with seasonal signals and US market holiday skipping.")

uploaded_file = st.file_uploader("Upload historical stock data (CSV/XLSX)", type=["csv", "xlsx"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file, engine='openpyxl')
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df.sort_values('Date', inplace=True)
    df.set_index('Date', inplace=True)

    st.subheader("📋 Data Preview")
    st.write(df.tail())

    df_recent = df[['Close']].copy().tail(1800)
    df_recent['day_sin']   = np.sin(2 * np.pi * df_recent.index.dayofweek / 7)
    df_recent['day_cos']   = np.cos(2 * np.pi * df_recent.index.dayofweek / 7)
    df_recent['month_sin'] = np.sin(2 * np.pi * df_recent.index.month / 12)
    df_recent['month_cos'] = np.cos(2 * np.pi * df_recent.index.month / 12)

    features = ['Close', 'day_sin', 'day_cos', 'month_sin', 'month_cos']
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_recent[features])

    def create_sequences(data, seq_len=60):
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i+seq_len])
            y.append(data[i+seq_len][0])
        return np.array(X), np.array(y)

    seq_len = 60
    X, y = create_sequences(scaled_data, seq_len)

    model = Sequential([
        Input(shape=(seq_len, X.shape[2])),
        LSTM(64),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    with st.spinner("⏳ Training LSTM..."):
        model.fit(X, y, epochs=20, batch_size=16, verbose=1, callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])
    st.success("✅ Model trained!")

    us_holidays = pd.to_datetime([
        "02-01-2023", "16-01-2023", "20-02-2023", "07-04-2023", "29-05-2023", "19-06-2023", "04-07-2023", "04-09-2023", "23-11-2023", "25-12-2023",
        "01-01-2024", "15-01-2024", "19-02-2024", "29-03-2024", "27-05-2024", "19-06-2024", "04-07-2024", "02-09-2024", "28-11-2024", "25-12-2024",
        "01-01-2025", "20-01-2025", "17-02-2025", "18-04-2025", "26-05-2025", "19-06-2025", "04-07-2025", "01-09-2025", "27-11-2025", "25-12-2025"
    ], format='%d-%m-%Y')

    historical_dates = pd.to_datetime(df.index.unique())
    def next_valid_trading_days(start, count):
        dates = []
        current = pd.Timestamp(start)
        limit = pd.Timestamp.max - pd.Timedelta(days=10)

        while len(dates) < count:
            current += pd.Timedelta(days=1)
            if current > limit:
                break  # safety stop to prevent infinite loop
            if current.weekday() < 5 and current not in us_holidays:
                dates.append(current)
        return dates

    def forecast_next_7(model, scaled_data, scaler, base_df):
        seq = scaled_data[-seq_len:].reshape(1, seq_len, -1)
        preds = []
        future_dates = next_valid_trading_days(base_df.index[-1], 7)

        for f_date in future_dates:
            pred_scaled = model.predict(seq, verbose=0)
            seasonal = [
                np.sin(2 * np.pi * f_date.weekday() / 7),
                np.cos(2 * np.pi * f_date.weekday() / 7),
                np.sin(2 * np.pi * f_date.month / 12),
                np.cos(2 * np.pi * f_date.month / 12)
            ]
            last_feats = seq[0, -1, 1:]
            full_scaled_row = np.concatenate([pred_scaled[0], last_feats])
            actual_pred = scaler.inverse_transform(full_scaled_row.reshape(1, -1))[0][0]
            preds.append(round(actual_pred, 2))

            new_row = np.concatenate([pred_scaled[0], seasonal])
            seq = np.append(seq[0][1:], [new_row], axis=0).reshape(1, seq_len, -1)

        return preds, future_dates

    predictions, forecast_dates = forecast_next_7(model, scaled_data, scaler, df_recent)

    st.subheader("📊 Predicted Closing Prices (Next Valid Trading Days)")
    for d, p in zip(forecast_dates, predictions):
        st.write(f"{d.strftime('%A, %d %b %Y')} → ${p}")

    st.subheader("📈 Forecast Plot")
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('#121212')
    ax.set_facecolor('#121212')
    ax.plot(forecast_dates, predictions, marker='o', linestyle='--', color='orange', label='Forecast')
    ax.set_title("7-Day Stock Price Forecast (Holiday-Aware)", color='white')
    ax.set_xlabel("Date", color='white')
    ax.set_ylabel("Price ($)", color='white')
    ax.tick_params(axis='x', rotation=45, colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.legend()
    st.pyplot(fig)
