import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import random

# Fetch data from Yahoo Finance
ticker = 'AAPL'  # Replace with the stock ticker you're interested in
start_date = '2023-01-01'
end_date = '2024-01-01'
df = yf.download(ticker, start=start_date, end=end_date)

# Drop missing values
df = df.dropna()

# Add technical indicators or features
df['MA_10'] = df['Close'].rolling(window=10).mean()
df['RSI'] = 100 - (100 / (1 + (df['Close'].diff().fillna(0) > 0).rolling(window=14).mean() / (df['Close'].diff().fillna(0) < 0).rolling(window=14).mean()))

# Define a simple sentiment score function for demonstration
def generate_sentiment_score():
    return random.uniform(-1, 1)

# Generate a column for sentiment scores
df['Sentiment_Score'] = [generate_sentiment_score() for _ in range(len(df))]

# Define features (including Sentiment_Score)
features = ['Open', 'High', 'Low', 'Volume', 'MA_10', 'RSI', 'Sentiment_Score']
target = 'Close'

# Drop rows with NaN values in technical indicators
df = df.dropna(subset=features)

# Feature scaling
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df[features])

# Target scaling
target_scaler = MinMaxScaler()
scaled_target = target_scaler.fit_transform(df[[target]])

# Convert to sequences for LSTM
def create_sequences(data, target, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = target[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

SEQ_LENGTH = min(10, len(df) - 1)
X, y = create_sequences(scaled_features, scaled_target, SEQ_LENGTH)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(SEQ_LENGTH, len(features)), return_sequences=True),
    Dropout(0.2),
    LSTM(50, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Predict on the test set
y_pred = model.predict(X_test)
y_test_inv = target_scaler.inverse_transform(y_test)
y_pred_inv = target_scaler.inverse_transform(y_pred)

# Calculate performance metrics
mae = mean_absolute_error(y_test_inv, y_pred_inv)
mse = mean_squared_error(y_test_inv, y_pred_inv)
rmse = np.sqrt(mse)

accuracy_percentage = 100 - (mae / np.mean(y_test_inv) * 100)

# Print metrics
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"Accuracy Percentage: {accuracy_percentage:.2f}%")

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(df.index[-len(y_test):], y_test_inv, label='Actual')
plt.plot(df.index[-len(y_test):], y_pred_inv, label='Predicted', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Predict future values considering sentiment
def predict_future_values_with_sentiment(model, last_sequence, future_steps, scaler):
    predictions = []
    current_sequence = last_sequence

    for _ in range(future_steps):
        # Reshape the sequence to match LSTM input shape (1, SEQ_LENGTH, features)
        current_sequence_reshaped = np.reshape(current_sequence, (1, SEQ_LENGTH, len(features)))

        # Predict the next value
        predicted_value = model.predict(current_sequence_reshaped)

        # Inverse transform the predicted value to original scale
        predicted_value_inv = scaler.inverse_transform(predicted_value)
        predictions.append(predicted_value_inv[0, 0])  # Append the predicted value

        # Generate a new sentiment score
        new_sentiment = generate_sentiment_score()

        # Create a new sequence with the predicted value
        new_feature_values = np.zeros((1, len(features)))  # Placeholder zeros for new features
        new_sequence = np.append(current_sequence[1:], new_feature_values, axis=0)

        # Update the sentiment score in the new sequence
        new_sequence[-1, -1] = new_sentiment  # Assuming last feature is sentiment

        # Update the last element in the new sequence with the predicted target (Close price)
        new_sequence[-1, -2] = predicted_value  # Assuming the second last column is the target (Close price)
        current_sequence = new_sequence

    return predictions

# Define the number of future steps to predict (30 days)
future_steps = 120

# Get the last sequence from the test data
last_sequence = X_test[-1]  # This is the last sequence the model saw

# Predict future values with sentiment
future_predictions = predict_future_values_with_sentiment(model, last_sequence, future_steps, target_scaler)

# Create a new range for the future predictions
future_dates = pd.date_range(start=df.index[-1], periods=future_steps + 1, freq='B')[1:]

# Plot actual, predicted, and future values
plt.figure(figsize=(12, 6))

# Plot the actual values for the test period
plt.plot(df.index[-len(y_test):], y_test_inv, label='Actual')

# Plot the predicted values for the test period
plt.plot(df.index[-len(y_test):], y_pred_inv, label='Predicted', linestyle='--')

# Plot the future predictions
plt.plot(future_dates, future_predictions, label='Future Predictions', linestyle='-.', color='orange')

plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.title(f'Stock Price Prediction for {ticker}')
plt.show()
