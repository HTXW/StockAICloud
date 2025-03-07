import yfinance as yf
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import matplotlib.dates as mdates
import joblib
import time

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


class CustomModel(tf.keras.Model):
    def __init__(self, num_layers, hidden_dim, kernel_size, output_dim):
        super(CustomModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.output_dim = output_dim
        self.cnn_blocks = [
            tf.keras.layers.Conv1D(hidden_dim, kernel_size, padding='causal', dilation_rate=2 ** i, activation='relu')
            for i in range(num_layers)]
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(output_dim)

    def call(self, inputs, training=False):
        x = inputs
        for cnn_block in self.cnn_blocks:
            x = cnn_block(x, training=training)
        x = self.flatten(x)
        output = self.dense(x)
        return output


def train_model(X_train, y_train, num_layers=4, hidden_dim=128, kernel_size=3, output_dim=1, learning_rate=0.01,
                epochs=300, batch_size=32):
    model = CustomModel(num_layers, hidden_dim, kernel_size, output_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.MeanSquaredError()

    @tf.function
    def train_step(batch_x, batch_y):
        with tf.GradientTape() as tape:
            logits = model(batch_x, training=True)
            loss = loss_fn(batch_y, logits)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return logits, loss

    start_time = time.time()

    pbar = tqdm(range(epochs), desc='train loop')
    for i in pbar:
        total_loss, total_acc = [], []
        for k in range(0, len(X_train), batch_size):
            batch_x = X_train[k:k + batch_size]
            batch_y = y_train[k:k + batch_size]
            logits, loss = train_step(batch_x, batch_y)
            total_loss.append(loss)
            total_acc.append(calculate_accuracy(batch_y, logits.numpy().flatten()))
        pbar.set_postfix(cost=np.mean(total_loss), acc=np.mean(total_acc))

    training_time = time.time() - start_time
    print(f'Traing time: {training_time:.2f} seconds')
    return model


def predict(model, X, batch_size=32):
    predictions = model(X, training=False)
    return predictions.numpy().flatten()


def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


def plot_predictions(dates, y_true, predictions, title):
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
    plt.show()


def calculate_accuracy(real, predict):
    real = np.array(real) + 1
    predict = np.array(predict) + 1
    percentage = 1 - np.sqrt(np.mean(np.square((real - predict) / real)))
    return percentage * 100


def main_process(column, num_layers=4, hidden_dim=128, kernel_size=3, learning_rate=0.01):
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
        model = train_model(X_train, y_train, num_layers=num_layers, hidden_dim=hidden_dim, kernel_size=kernel_size,
                            learning_rate=learning_rate)

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

    # Save the model and scaler
    model.save(f'model_{column.lower()}.h5')
    joblib.dump(scaler, f'scaler_{column.lower()}.pkl')

    plot_predictions(train_data['Date'].values[-len(y_train):],
                     scaler.inverse_transform(y_train.reshape(-1, 1)).flatten(),
                     [scaler.inverse_transform(result['train_predictions'].reshape(-1, 1)).flatten() for result in
                      simulation_results], f'{column} Training Data Predictions (All Simulations)')
    plot_predictions(test_data['Date'].values[-len(y_test):], scaler.inverse_transform(y_test.reshape(-1, 1)).flatten(),
                     [scaler.inverse_transform(result['test_predictions'].reshape(-1, 1)).flatten() for result in
                      simulation_results], f'{column} Testing Data Predictions (All Simulations)')

    return model, scaler


model_close, scaler_close = main_process('Close', num_layers=4, hidden_dim=128, kernel_size=3, learning_rate=0.01)
model_open, scaler_open = main_process('Open', num_layers=4, hidden_dim=128, kernel_size=3, learning_rate=0.01)