import os
import ccxt
import numpy as np
import pandas as pd
from ta import add_all_ta_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump, load
import requests
from textblob import TextBlob
import logging
from dotenv import load_dotenv
from datetime import datetime
import time

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CommandHandler, ContextTypes, CallbackQueryHandler


load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Binance API setup
exchange = ccxt.binance(
    {
        "apiKey": os.getenv("BINANCE_API_KEY"),  # get apis from binance testnet
        "secret": os.getenv("BINANCE_API_SECRET"),
        "options": {"defaultType": "future"},
    }
)
# exchange.set_sandbox_mode(True)  # for testing, disable in prod


model_path = "ml_model.joblib"
training_count = 0
trailing_stop_loss_pct = 0.02
take_profit_ratio = 0.05
top_pairs = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT"]


# sentiments??


def fetch_market_data(pair, timeframe="1h", limit=100):
    """Fetch historical OHLCV data for the given trading pair."""
    try:
        ohlcv = exchange.fetch_ohlcv(pair, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        if df.empty:
            logging.warning(f"No data returned for {pair}.")
        return df
    except Exception as e:
        logging.error(f"Error fetching market data for {pair}: {e}")
        return pd.DataFrame()


def compute_technical_indicators(data):
    """Compute RSI, Bollinger Bands, and SMA."""
    data = add_all_ta_features(
        data, open="open", high="high", low="low", close="close", volume="volume"
    )
    data = data[
        [
            "close",
            "volume",
            "volatility_bbm",
            "volatility_bbh",
            "volatility_bbl",
            "trend_sma_slow",
            "momentum_rsi",
        ]
    ]
    data.rename(
        columns={
            "volatility_bbm": "bb_middle",
            "volatility_bbh": "bb_upper",
            "volatility_bbl": "bb_lower",
            "trend_sma_slow": "sma",
            "momentum_rsi": "rsi",
        },
        inplace=True,
    )
    return data


def get_sentiment(text):
    """Calculate sentiment polarity from text."""
    analysis = TextBlob(text)
    return analysis.sentiment.polarity


def fetch_latest_sentiment():
    """Fetch sentiment scores from real-time news or social media."""
    try:
        news_response = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": "crypto OR bitcoin OR ethereum",
                "sortBy": "publishedAt",
                "apiKey": os.getenv("NEWS_API_KEY"),
            },
        )
        if news_response.status_code == 200:
            articles = news_response.json().get("articles", [])
            if not articles:
                logging.warning("No articles found for sentiment analysis.")
            all_text = " ".join([article["title"] for article in articles])
            return get_sentiment(all_text)
        else:
            logging.warning(f"News API error: {news_response.status_code}")
            return 0
    except Exception as e:
        logging.error(f"Error fetching real-time sentiment: {e}")
        return 0


# --- ML Functions ---
def train_or_load_model(data, retrain_interval=100):
    """Train or load the ML model with periodic retraining."""
    global training_count
    training_count = training_count + 1 if "training_count" in globals() else 0

    if not os.path.exists(model_path) or training_count >= retrain_interval:
        if data.empty:
            logging.warning("No data available for training the model.")
            return None
        X = data[["rsi", "bb_upper", "bb_lower", "sma"]].values
        y = np.where(data["close"].shift(-1) > data["close"], 1, 0)
        model = RandomForestClassifier()
        model.fit(X[:-1], y[:-1])
        dump(model, model_path)
        logging.info("ML model trained and saved.")
        training_count = 0
    else:
        model = load(model_path)
        logging.info("ML model loaded.")
    return model


def predict_price_movement(model, data):
    """Predict whether the price will increase or decrease."""
    X = data[["rsi", "bb_upper", "bb_lower", "sma"]].values[-1:]
    prediction = model.predict(X)
    return prediction[0]


# --- Trade Execution ---
# def place_order(order_type, pair, amount):
#     """Place buy or sell orders securely."""
#     try:
#         if order_type == "buy":
#             order = exchange.create_market_buy_order(pair, amount)
#         elif order_type == "sell":
#             order = exchange.create_market_sell_order(pair, amount)
#         else:
#             logging.error(f"Invalid order type: {order_type}")
#             return None
#         logging.info(f"{order_type.capitalize()} order placed: {order}")
#         return order
#     except Exception as e:
#         logging.error(f"Error placing {order_type} order for {pair}: {e}")
#         return None


def trailing_stop_loss(entry_price, current_price):
    """Calculate the trailing stop-loss price."""
    stop_loss_price = entry_price * (1 - trailing_stop_loss_pct)
    return max(stop_loss_price, current_price * (1 - trailing_stop_loss_pct))


def main():
    while True:  
        # Fetch market data for all pairs
        dataframes = {
            pair: compute_technical_indicators(fetch_market_data(pair))
            for pair in top_pairs
        }

        # Filter out empty DataFrames
        dataframes = {pair: df for pair, df in dataframes.items() if not df.empty}

        if not dataframes:
            logging.warning("No valid market data available. Retrying in 1 minute.")
            time.sleep(60)  # Wait before retrying
            continue

        for pair, data in dataframes.items():
            # Ensure there are enough data points to calculate percentage change
            if len(data) < 2:
                logging.warning(f"Not enough data for {pair}. Skipping.")
                continue

            # Calculate percentage change
            recent_close = data["close"].iloc[-1]
            previous_close = data["close"].iloc[-2]
            percentage_change = ((recent_close - previous_close) / previous_close) * 100

            if percentage_change >= 5:
                logging.info(
                    f"🚀 {pair}: Price rose by {percentage_change:.2f}%! Recent close: {recent_close}, Previous close: {previous_close}"
                )
            else:
                logging.info(
                    f"{pair}: Price change is {percentage_change:.2f}%, below the 5% threshold."
                )

        
        time.sleep(600)  

# Run the bot
if __name__ == "__main__":
    main()
