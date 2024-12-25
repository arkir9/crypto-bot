import unittest
import pandas as pd
from main import fetch_market_data, compute_technical_indicators, train_or_load_model

class TestCryptoBot(unittest.TestCase):

    def test_fetch_market_data(self):
        # Test fetching market data for a valid pair
        data = fetch_market_data("BTC/USDT", "1h", limit=10)
        self.assertFalse(data.empty, "Market data should not be empty for valid pair.")

    def test_apply_technical_indicators(self):
        # Mock data simulating fetched market data with sufficient rows
        sample_data = pd.DataFrame({
            "open": [100, 102, 101, 105, 107, 110, 108, 109, 111, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125],
            "high": [102, 103, 102, 106, 108, 111, 109, 110, 112, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126],
            "low": [99, 101, 100, 104, 106, 109, 107, 108, 110, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124],
            "close": [100, 102, 101, 105, 107, 110, 108, 109, 111, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125],
            "volume": [10, 15, 10, 20, 25, 30, 20, 15, 10, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
        })
        
        # Call the function to compute technical indicators
        indicators = compute_technical_indicators(sample_data)
        
        # Check if the expected columns are present in the output DataFrame
        self.assertIn("rsi", indicators.columns, "RSI should be calculated and added to DataFrame.")
        self.assertIn("sma", indicators.columns, "SMA should be calculated and added to DataFrame.")
        self.assertIn("bb_middle", indicators.columns, "Bollinger Bands middle should be calculated and added to DataFrame.")
        self.assertIn("bb_upper", indicators.columns, "Bollinger Bands upper should be calculated and added to DataFrame.")
        self.assertIn("bb_lower", indicators.columns, "Bollinger Bands lower should be calculated and added to DataFrame.")

    def test_train_or_load_model(self):
        # Mock data for model training
        sample_data = pd.DataFrame({
            "rsi": [30, 40, 50, 60, 70],
            "bb_upper": [110, 112, 114, 116, 118],
            "bb_lower": [90, 88, 86, 84, 82],
            "sma": [100, 101, 102, 103, 104],
            "close": [100, 102, 101, 105, 107],
        })
        
        model = train_or_load_model(sample_data)
        self.assertIsNotNone(model, "Model should be trained or loaded successfully.")

if __name__ == "__main__":
    unittest.main()
