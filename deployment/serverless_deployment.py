from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import logging
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from mangum import Mangum

app = FastAPI()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def custom_mse(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)


try:
    model_close = tf.keras.models.load_model('app/models/model_close.h5', custom_objects={'mse': custom_mse})
    model_open = tf.keras.models.load_model('app/models/model_open.h5', custom_objects={'mse': custom_mse})
    scaler_close = joblib.load('app/scalers/scaler_close.pkl')
    scaler_open = joblib.load('app/scalers/scaler_open.pkl')
    logger.info("Models and scalers loaded successfully.")
except Exception as e:
    logger.error(f"Error loading models or scalers: {e}")
    raise

class StockRequest(BaseModel):
    date: str

def fetch_real_time_data(stock_symbol):
    try:
        data = yf.download(stock_symbol, period="1y", interval="1d")
        data.reset_index(inplace=True)
        data.rename(columns={
            'Date': 'Date',
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Close',
            'Adj Close': 'Adj Close',
            'Volume': 'Volume'
        }, inplace=True)
        return data
    except Exception as e:
        logger.error(f"Error fetching real-time data: {e}")
        raise

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI application"}

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("app/favicon.ico")

@app.post("/predict")
async def predict(stock_request: StockRequest):
    date_str = stock_request.date
    stock_symbol = "HDFCBANK.NS"

    logger.info(f"Received request with date: {date_str}")

    try:
        date = datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        logger.error("Invalid date format")
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    if date.weekday() >= 5:
        logger.info("Entered date is a non-working day (Saturday or Sunday)")
        return {
            "date": date_str,
            "message": "The entered date falls on a non-working day (Saturday or Sunday). Please enter a weekday."
        }

    try:
        data = fetch_real_time_data(stock_symbol)
        logger.info(f"Fetched real-time data for {stock_symbol}")

        predicted_close_price, close_test_accuracy = predict_for_date(model_close, scaler_close, data, date, 'Close')
        logger.info(f"Predicted close price: {predicted_close_price}, Accuracy: {close_test_accuracy}")

        predicted_open_price, open_test_accuracy = predict_for_date(model_open, scaler_open, data, date, 'Open')
        logger.info(f"Predicted open price: {predicted_open_price}, Accuracy: {open_test_accuracy}")

        average_test_accuracy = (close_test_accuracy + open_test_accuracy) / 2

        response = {
            "date": date_str,
            "predicted_open_price": round(float(predicted_open_price), 2),
            "predicted_close_price": round(float(predicted_close_price), 2),
            "average_test_accuracy": round(average_test_accuracy, 2),
            "disclaimer": f"The stock price predictions provided by this model, with an accuracy rate of {round(average_test_accuracy, 2)}%, are based on historical data and advanced algorithms. However, actual market prices may vary due to unforeseen factors. Please use this information as one of many tools in your decision-making process and consult financial professionals when needed. This is not financial advice."
        }
        logger.info(f"Response: {response}")
        return response
    except ValueError as e:
        logger.error(f"Error in prediction: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Internal Server Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

def predict_for_date(model, scaler, data, target_date, column):
    try:
        latest_date = data['Date'].max()

        if target_date <= latest_date:
            raise ValueError(f"The date {target_date} is not in the future. Please enter a future date.")

        if target_date > latest_date + timedelta(days=7):
            raise ValueError(f"The date {target_date} is beyond the 7-day prediction range. Please enter a date within the 7-day prediction range.")

        days_ahead = (target_date - latest_date).days
        sequence = data.iloc[-30:][column].values.reshape(-1, 1)
        sequence_scaled = scaler.transform(sequence).reshape(1, 30, 1)

        for i in range(days_ahead):
            prediction_scaled = model.predict(sequence_scaled).flatten()
            new_prediction = scaler.inverse_transform(prediction_scaled.reshape(-1, 1)).flatten()[0]
            sequence = np.roll(sequence, -1)
            sequence[-1] = new_prediction
            sequence_scaled = scaler.transform(sequence).reshape(1, 30, 1)

        test_accuracy = calculate_accuracy(data.iloc[-30:][column].values, scaler.inverse_transform(sequence_scaled.flatten().reshape(-1, 1)))
        return new_prediction, test_accuracy
    except Exception as e:
        logger.error(f"Error in prediction for date {target_date}: {e}")
        raise

def calculate_accuracy(y_true, y_pred):
    y_true = np.array(y_true).reshape(-1, 1) + 1
    y_pred = np.array(y_pred).reshape(-1, 1) + 1
    percentage = 1 - np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))
    return percentage * 100

handler = Mangum(app)