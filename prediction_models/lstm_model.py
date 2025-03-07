import time
import yfinance as yf
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.dates as mdates
import joblib

tf.compat.v1.random.set_random_seed(1234)

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

ticker = 'HDFCBANK.NS'
period = '2y'

stock_data = yf.download(ticker, period=period)
csv_file_path = 'HDFCBANK_STOCK_DATA.csv'
stock_data.to_csv(csv_file_path)

stock_data = pd.read_csv(csv_file_path, skiprows=3, header=None)
stock_data.columns = ["Date", "Adj Close", "Close", "High", "Low", "Open", "Volume"]
stock_data['Date'] = pd.to_datetime(stock_data['Date'])
stock_data.sort_values('Date', inplace=True)


def prepare_data(stock_data, column):
    data = stock_data[['Date', column]]
    split_index = int(len(data) * 0.8)
    train_data = data.iloc[:split_index]
    test_data = data.iloc[split_index:]
    scaler = MinMaxScaler()
    train_data[column] = scaler.fit_transform(train_data[[column]])
    test_data[column] = scaler.transform(test_data[[column]])
    return train_data, test_data, scaler


def create_sequences(data, timestamp, column):
    X = []
    y = []
    for i in range(len(data) - timestamp):
        X.append(data.iloc[i:i + timestamp][column].values)
        y.append(data.iloc[i + timestamp][column])
    return np.array(X), np.array(y)


def train_model(X_train, y_train, num_layers=1, units=128, dropout_rate=0.8, output_dim=1, learning_rate=0.01,
                epochs=300, batch_size=32):
    model = tf.keras.Sequential()
    for i in range(num_layers):
        model.add(LSTM(units, return_sequences=(i < num_layers - 1)))
        model.add(Dropout(dropout_rate))
    model.add(Dense(output_dim))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    return model


def predict(model, X, batch_size=32):
    predictions = model.predict(X, batch_size=batch_size)
    return predictions.flatten()


def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


def plot_predictions(dates, y_true, predictions, title, save_path=None):
    plt.figure(figsize=(11, 5))
    plt.plot(dates, y_true, label='True Price', color='black', linewidth=2)
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan', 'magenta']
    for i, prediction in enumerate(predictions):
        plt.plot(dates[-len(prediction):], prediction, label=f'Simulation {i + 1}', alpha=0.5,
                 color=colors[i % len(colors)])
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.gcf().autofmt_xdate()

    if save_path:
        plt.savefig(save_path)

    plt.show()


def calculate_accuracy(real, predict):
    real = np.array(real) + 1
    predict = np.array(predict) + 1
    percentage = 1 - np.sqrt(np.mean(np.square((real - predict) / real)))
    return percentage * 100


def main_process(column, num_layers=2, units=256, dropout_rate=0.5, learning_rate=0.001):
    train_data, test_data, scaler = prepare_data(stock_data, column)
    timestamp = 10

    X_train, y_train = create_sequences(train_data, timestamp, column)
    X_test, y_test = create_sequences(test_data, timestamp, column)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    simulation_size = 6
    simulation_results = []

    for sim in range(simulation_size):
        print(f"\nRunning simulation {sim + 1}/{simulation_size}")
        model = train_model(X_train, y_train, num_layers=num_layers, units=units, dropout_rate=dropout_rate,
                            learning_rate=learning_rate)

        inference_start_time = time.time()
        test_predictions = predict(model, X_test)
        inference_end_time = time.time()

        total_inference_time = inference_end_time - inference_start_time
        avg_inference_per_sample = total_inference_time / len(X_test)

        print(f"Inference time for test set (simulation {sim + 1}): {total_inference_time:.4f} seconds")
        print(f"Average inference time per sample: {avg_inference_per_sample:.6f} seconds/sample")

        train_predictions = predict(model, X_train)
        test_predictions = predict(model, X_test)

        train_accuracy = calculate_accuracy(y_train, train_predictions)
        test_accuracy = calculate_accuracy(y_test, test_predictions)
        rmse, mae, r2 = calculate_metrics(y_test, test_predictions)

        simulation_results.append({
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'train_predictions': train_predictions,
            'test_predictions': test_predictions,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        })

        print(f'Training Accuracy: {train_accuracy:.2f}%')
        print(f'Testing Accuracy: {test_accuracy:.2f}%')
        print(f'RMSE: {rmse:.4f}')
        print(f'MAE: {mae:.4f}')
        print(f'R2: {r2:.4f}')

    avg_train_accuracy = np.mean([result['train_accuracy'] for result in simulation_results])
    avg_test_accuracy = np.mean([result['test_accuracy'] for result in simulation_results])
    avg_rmse = np.mean([result['rmse'] for result in simulation_results])
    avg_mae = np.mean([result['mae'] for result in simulation_results])
    avg_r2 = np.mean([result['r2'] for result in simulation_results])

    print(f"\nAverage Training Accuracy for {column}: {avg_train_accuracy:.2f}%")
    print(f"Average Testing Accuracy for {column}: {avg_test_accuracy:.2f}%")
    print(f"Average RMSE for {column}: {avg_rmse:.4f}")
    print(f"Average MAE for {column}: {avg_mae:.4f}")
    print(f"Average R2 for {column}: {avg_r2:.4f}")

    model.save(f'model_{column.lower()}.h5')
    joblib.dump(scaler, f'scaler_{column.lower()}.pkl')

    plot_predictions(
        train_data['Date'].values[-len(y_train):],
        scaler.inverse_transform(y_train.reshape(-1, 1)).flatten(),
        [scaler.inverse_transform(result['train_predictions'].reshape(-1, 1)).flatten() for result in
         simulation_results],
        f'{column} Training Data Predictions (All Simulations)',
        save_path=f'{column}_training_predictions.png'
    )

    plot_predictions(
        test_data['Date'].values[-len(y_test):],
        scaler.inverse_transform(y_test.reshape(-1, 1)).flatten(),
        [scaler.inverse_transform(result['test_predictions'].reshape(-1, 1)).flatten() for result in
         simulation_results],
        f'{column} Testing Data Predictions (All Simulations)',
        save_path=f'{column}_testing_predictions.png'
    )

    return model, scaler


model_close, scaler_close = main_process('Close', num_layers=2, units=256, dropout_rate=0.5, learning_rate=0.001)
model_open, scaler_open = main_process('Open', num_layers=2, units=256, dropout_rate=0.5, learning_rate=0.001)