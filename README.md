This project predicts stock prices using Long Short-Term Memory (LSTM) models and sentiment analysis , leveraging historical stock data and social sentiment to enhance prediction accuracy.

### Overview

Stock prices are challenging to predict due to market volatility and the influence of both historical data and external sentiment. 
This project combines financial data with sentiment analysis, using LSTM models for time-series predictions and BERT for real-time sentiment scoring from news and social media.

### Features

Data Collection: Gathers real-time stock data using yfinance and web scraping.
Technical Indicators: Calculates Moving Averages (MA), Relative Strength Index (RSI), and other indicators.
Sentiment Analysis: Uses BERT to assign sentiment scores from relevant news articles and tweets.
Prediction Model: LSTM model trained on historical stock data, sentiment scores, and technical indicators.
Visualization: Graphs to visualize actual vs. predicted values and 30-day future price predictions.
Project Structure

### Data Collection
Real-time stock data is obtained from Yahoo Finance via yfinance. Sentiment data is collected through scraping financial news websites and social media for mentions of selected stocks. Sentiment analysis is performed using a BERT-based model.

### Normalisation and feature engineering

Feature Engineering: Technical indicators (e.g., MA, RSI) are calculated and added as features.
Sentiment Score Integration: Sentiment scores are added to represent market sentiment.

### Model Training
LSTM Model: Trained on time-series data with additional features (technical indicators, sentiment scores).
Training Process: Sequential training with parameter tuning to optimize performance.
Hyperparameter Tuning: Batch size, learning rate, and number of epochs adjusted for accuracy.

### Evaluation and Results
Graphs display actual vs. predicted stock prices for observed periods and 30-day future predictions. Evaluation metrics include Mean Absolute Error (MAE) and Mean Squared Error (MSE).

### Future Work

Enhanced Feature Engineering: Adding features like market volatility and sector-wise sentiment.
Refinement of Sentiment Analysis: Expanding sources for more granular sentiment.
Model Tuning: Experimenting with GRU and Transformer models for improved results.

### Requirements

Python 3.x
Libraries: yfinance, pandas, numpy, tensorflow, transformers, beautifulsoup4, and matplotlib

### License

This project is licensed under the MIT License.
