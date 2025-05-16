import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from math import sqrt
import tensorflow as tf
import numpy as np
import random

# Set seed values for reproducibility
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

# Companies list
companies = ['AAPL', 'NVDA', 'LMT']  # Apple, Nvidia, Lockheed Martin

# Prepare a dictionary to store results for each company
results = {}


# Define function to fetch data and run prediction
# Load sentiment CSV
sentiment_df = pd.read_csv(r'E:\Pycharm\basicreview\sentiment_data.csv')  # Update path as necessary

# Define sentiment score mapping
sentiment_mapping = {
    "positive": 1,
    "neutral": 0,
    "negative": -1
}

# Process stock data
def run_stock_prediction(ticker):
    # Fetch stock data
    df = yf.download(ticker, start="2020-01-01", end="2023-01-01")
    df = df[['Close']]

    # Reset the index to avoid 'Date' being both index and column
    df = df.reset_index()

    # Ensure 'Date' is in datetime64 format
    df['Date'] = pd.to_datetime(df['Date'])

    # Merge stock data with sentiment data
    company_sentiments = sentiment_df[sentiment_df['Company'] == ticker]

    # Ensure 'Date' in sentiment data is also datetime64 format
    company_sentiments['Date'] = pd.to_datetime(company_sentiments['Date'])

    # Map sentiment to numerical values
    company_sentiments.loc[:, 'Sentiment'] = company_sentiments['Sentiment'].map(sentiment_mapping)

    # Merge with main dataframe on date
    df = df.merge(company_sentiments[['Date', 'Sentiment']], on='Date', how='left')

    # Fill missing sentiment values as neutral (0)
    df['Sentiment'].fillna(0, inplace=True)

    # Scale sentiment impact
    sentiment_weight = 0.2  # Adjust influence on prediction
    df['Sentiment'] = df['Sentiment'] * sentiment_weight


    # (Rest of the code remains the same)


    def prepare_data(data, time_step=60):
        X, y = [], []
        for i in range(len(data) - time_step - 1):
            X.append(data[i:(i + time_step), :])
            y.append(data[i + time_step, 0])  # Predict based on 'Close' price
        return np.array(X), np.array(y)

    time_step = 60
    # Prepare data for LSTM with adjusted sentiment
    features = df[['Close', 'Sentiment']].values
    X, y = prepare_data(features, time_step)

    # Data Preprocessing
    scaler = MinMaxScaler(feature_range=(0, 1))
    df[['Close', 'Sentiment']] = scaler.fit_transform(df[['Close', 'Sentiment']])

    # Prepare data for LSTM

    features = df[['Close', 'Sentiment']].values
    X, y = prepare_data(features, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 2)

    # Split into training and testing sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Build LSTM model with additional dropout layers for regularization
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 2)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Define learning rate scheduler
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1)

    # Train the model with more epochs and learning rate scheduler
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1, callbacks=[lr_scheduler],
              validation_data=(X_test, y_test))

    # Predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Inverse transform predictions
    train_predict = scaler.inverse_transform(np.concatenate((train_predict, np.zeros_like(train_predict)), axis=1))[:,
                    0]
    y_train = scaler.inverse_transform(
        np.concatenate((y_train.reshape(-1, 1), np.zeros_like(y_train.reshape(-1, 1))), axis=1))[:, 0]
    test_predict = scaler.inverse_transform(np.concatenate((test_predict, np.zeros_like(test_predict)), axis=1))[:, 0]
    y_test = scaler.inverse_transform(
        np.concatenate((y_test.reshape(-1, 1), np.zeros_like(y_test.reshape(-1, 1))), axis=1))[:, 0]

    # Generate predictions for the next 30 days
    # Generate predictions for the next 30 days
    last_60_days = df[['Close', 'Sentiment']][-60:].values
    pred_input = last_60_days.reshape(1, -1, 2)
    predictions = []

    for _ in range(30):  # Predict next 30 days
        pred = model.predict(pred_input)
        predictions.append(pred[0][0])
        # Apply a slight sentiment-based shift (scaled by sentiment_weight)
        sentiment_variation = sentiment_weight * np.random.randn()  # Slight random sentiment shift
        next_input = np.array([[pred[0][0], pred_input[0, -1, 1] + sentiment_variation]])  # Update sentiment
        pred_input = np.append(pred_input[:, 1:, :], [next_input], axis=1)

    predictions = scaler.inverse_transform(
        np.concatenate((np.array(predictions).reshape(-1, 1), np.zeros_like(np.array(predictions).reshape(-1, 1))),
                       axis=1))[:, 0]

    # Store results in dictionary
    results[ticker] = {
        'train_predict': train_predict,
        'y_train': y_train,
        'test_predict': test_predict,
        'y_test': y_test,
        'predictions': predictions
    }

    # --- Graphs ---
    # Graph 1: Actual vs Predicted (Training and Testing Data)


    # Graph 1: Actual vs Predicted (Training and Testing Data)
    # Graph 1: Actual vs Predicted (Training and Testing Data)
    plt.figure(figsize=(12, 6))

    # Actual values should span from the start to the full length (training + testing data)
    plt.plot(np.arange(len(y_train) + len(y_test)), np.concatenate((y_train, y_test)), label='Actual Price',
             color='black')

    # Train predictions plot from 0 to len(train_predict)
    plt.plot(np.arange(len(train_predict)), train_predict, label='Train Predictions', color='green')

    # Test predictions plot, shifted 60 units to the left, aligning it with actual test data
    plt.plot(np.arange(len(train_predict), len(train_predict) + len(test_predict)), test_predict,
             label='Test Predictions', color='red')

    plt.title(f"{ticker} Actual vs Predicted Price (Training and Testing Data)")
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.show()

    # Graph 2: Predicted Price for the Next 30 Days
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 31), predictions, label="Predicted Price (Next 30 Days)", color='blue')
    plt.title(f"{ticker} Predicted Price for the Next 30 Days")
    plt.xlabel("Days")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.show()

    # Graph 3: Predicted Price Range (Upper and Lower Bounds)
    plt.figure(figsize=(12, 6))

    # Assuming we already have predictions from the previous steps
    predicted_values = np.array(results[ticker]['predictions'])

    # Dynamically adjust bounds based on the range of predicted values
    sentiment_influence = 0.05 * predicted_values  # Adjust this multiplier based on sentiment impact
    upper_bound = predicted_values + sentiment_influence
    lower_bound = predicted_values - sentiment_influence

    # Plot the range with proper scaling
    plt.fill_between(range(1, 31), lower_bound, upper_bound, color='gray', alpha=0.3, label="Predicted Range")
    plt.plot(range(1, 31), predicted_values, color='blue', label="Predicted Price (Next 30 Days)")

    plt.title(f"{ticker} Predicted Price Range for the Next 30 Days")
    plt.xlabel("Days")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


# Run prediction for each company
for company in companies:
    run_stock_prediction(company)

# Summarize errors
from sklearn.metrics import r2_score

# Summarize errors with additional metrics
for ticker in results:
    y_test = results[ticker]['y_test']
    test_predict = results[ticker]['test_predict']

    mse = mean_squared_error(y_test, test_predict)
    rmse = sqrt(mse)
    mae = mean_absolute_error(y_test, test_predict)

    # Additional metrics
    r2 = r2_score(y_test, test_predict)
    mape = np.mean(np.abs((y_test - test_predict) / y_test)) * 100  # Convert to percentage
    smape = 100 * np.mean(2 * np.abs(y_test - test_predict) / (np.abs(y_test) + np.abs(test_predict)))
    mbe = np.mean(test_predict - y_test)  # Mean Bias Error

    # Display metrics
    print(f"Results for {ticker}:")
    print(f"  MSE = {mse}")
    print(f"  RMSE = {rmse}")
    print(f"  MAE = {mae}")
    print(f"  R² Score = {r2}")
    print(f"  MAPE = {mape}%")
    print(f"  SMAPE = {smape}%")
    print(f"  MBE = {mbe}")
    print("-" * 40)
