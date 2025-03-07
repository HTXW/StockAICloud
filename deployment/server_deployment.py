from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
import io
import base64
import os
import csv

app = FastAPI()


def custom_mse(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)


model_close = tf.keras.models.load_model('model_close.h5', custom_objects={'mse': custom_mse})
model_open = tf.keras.models.load_model('model_open.h5', custom_objects={'mse': custom_mse})
scaler_close = joblib.load('scaler_close.pkl')
scaler_open = joblib.load('scaler_open.pkl')

csv_file_path = 'HDFCBANK_STOCK_DATA.csv'
if not os.path.exists(csv_file_path):
    raise FileNotFoundError(f"{csv_file_path} not found.")

try:

    with open(csv_file_path, 'r') as infile, open('preprocessed_file.csv', 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        for row in reader:
            writer.writerow(row[0].split(','))

    stock_data = pd.read_csv('preprocessed_file.csv')

    expected_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    if not all(col in stock_data.columns for col in expected_columns):
        raise KeyError(f"CSV file is missing one or more expected columns: {expected_columns}")

    stock_data['Date'] = pd.to_datetime(stock_data['Date'])

    print("CSV file loaded successfully. Columns:", stock_data.columns)
    print("First few rows of data:")
    print(stock_data.head())

except Exception as e:
    raise Exception(f"Error loading or processing CSV file: {e}")


class StockRequest(BaseModel):
    date: str


@app.post("/predict")
async def predict(stock_request: StockRequest):
    date_str = stock_request.date
    try:
        date = datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    try:
        predicted_close_price = predict_for_date(model_close, scaler_close, stock_data, date, 'Close')
        predicted_open_price = predict_for_date(model_open, scaler_open, stock_data, date, 'Open')

        stock_data['Close_MA_7'] = stock_data['Close'].rolling(window=7).mean()

        img = io.BytesIO()
        plt.figure(figsize=(11, 5))
        plt.plot(stock_data['Date'], stock_data['Close'], label='Actual Close Price', color='black')
        plt.plot(stock_data['Date'], stock_data['Close_MA_7'], label='7-Day Moving Average', color='blue')
        plt.title('7-Day Moving Average of HDFC Bank Close Prices')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        graph_url = base64.b64encode(img.getvalue()).decode()

        return {
            "date": date_str,
            "predicted_open_price": predicted_open_price,
            "predicted_close_price": predicted_close_price,
            "moving_average_graph": f"data:image/png;base64,{graph_url}"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


def predict_for_date(model, scaler, data, date, column):
    latest_date = data['Date'].max()

    if date <= latest_date:
        raise ValueError(f"The date {date} is not in the future. Please enter a future date.")

    if date > latest_date + timedelta(days=7):
        raise ValueError(f"The date {date} is beyond the 7-day prediction range. Please enter a date within the next 7 >

        sequence = data.iloc[-5:][column].values.reshape(-1, 1)
        sequence_scaled = scaler.transform(sequence).reshape(1, 5, 1)
        prediction_scaled = model.predict(sequence_scaled).flatten()
        prediction = scaler.inverse_transform(prediction_scaled.reshape(-1, 1)).flatten()

    return prediction[0]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)