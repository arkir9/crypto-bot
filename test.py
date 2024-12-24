import logging
from datetime import datetime
from main import fetch_market_data, train_or_load_model, get_sentiment
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.trend import SMAIndicator

# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Function to apply technical indicators (RSI, Bollinger Bands, SMA)
def apply_technical_indicators(data):
    """
    Apply technical indicators (RSI, Bollinger Bands, SMA) to the market data.

    Parameters:
        data (DataFrame): DataFrame with 'close' column for calculations.

    Returns:
        DataFrame: Updated DataFrame with new indicator columns.
    """
    # Calculate RSI
    rsi = RSIIndicator(close=data["close"], window=14)
    data["rsi"] = rsi.rsi()

    # Calculate Bollinger Bands
    bb = BollingerBands(close=data["close"], window=20, window_dev=2)
    data["bb_upper"] = bb.bollinger_hband()
    data["bb_lower"] = bb.bollinger_lband()

    # Calculate SMA
    sma = SMAIndicator(close=data["close"], window=50)
    data["sma"] = sma.sma_indicator()

    return data


# Testing script
def test_bot():
    logging.info("Starting Bot Testing Script...")

    # Test 1: Market Data Fetching
    logging.info("TEST 1: Fetching Market Data...")
    try:
        symbol = "BTC/USDT"
        data = fetch_market_data(symbol, "1h", limit=10)
        if not data.empty:
            logging.info(f"Market data fetched successfully for {symbol}.")
            logging.info(data.head())
        else:
            logging.warning("Market data is empty!")
    except Exception as e:
        logging.error(f"Error during market data fetching: {e}")

    # Test 2: Technical Indicators
    logging.info("TEST 2: Applying Technical Indicators...")
    try:
        if not data.empty:
            indicators = apply_technical_indicators(data)
            logging.info("Technical indicators applied successfully.")
            logging.info(indicators.tail())
        else:
            logging.warning("Skipping indicators test due to empty market data.")
    except Exception as e:
        logging.error(f"Error during indicator calculation: {e}")

    # Test 3: Machine Learning Model
    logging.info("TEST 3: ML Model Training or Loading...")
    try:
        if not data.empty:
            model = train_or_load_model(data)
            latest_data = data.iloc[-1]
            features = latest_data[
                ["rsi", "bb_upper", "bb_lower", "sma"]
            ].values.reshape(1, -1)
            prediction = model.predict(features)[0]
            logging.info(
                f"Prediction for the next price movement: {prediction} (1 = Up, 0 = Down)"
            )
        else:
            logging.warning("Skipping ML test due to empty market data.")
    except Exception as e:
        logging.error(f"Error during ML model testing: {e}")

    # Test 4: Sentiment Analysis
    logging.info("TEST 4: Sentiment Analysis...")
    try:
        example_text = (
            "Bitcoin is surging in popularity due to positive market conditions."
        )
        sentiment_score = get_sentiment(example_text)
        logging.info(f"Sentiment analysis score for test input: {sentiment_score}")
    except Exception as e:
        logging.error(f"Error during sentiment analysis: {e}")

    # Test 5: Trade Execution (Dry Run)
    logging.info("TEST 5: Simulated Trade Execution (Dry Run)...")
    try:
        order_type = "buy"
        pair = "BTC/USDT"
        amount = 0.001
        logging.info(f"Dry Run: Would place {order_type} order for {amount} of {pair}.")
        # Uncomment the following line to test real execution (use with caution!)
        # place_order(order_type, pair, amount)
    except Exception as e:
        logging.error(f"Error during simulated trade execution: {e}")

    # Conclusion
    logging.info("Testing Completed. Review logs for any issues.")


# Run the testing script
if __name__ == "__main__":
    test_bot()
