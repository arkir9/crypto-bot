import os
import logging
import time
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from dotenv import load_dotenv
import pandas as pd
import ccxt
import asyncio
import nest_asyncio

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

# Binance API setup
exchange = ccxt.binance(
    {
        "apiKey": os.getenv("BINANCE_API_KEY"),
        "secret": os.getenv("BINANCE_API_SECRET"),
    }
)

# Global variables
top_pairs = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT"]
last_notified_price = {}  # To track previous prices for notifications
user_preferences = {}  # Global variable to store user preferences


# Fetch market data
def fetch_market_data(pair, timeframe="1h", limit=2):
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


# Telegram bot handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a welcome message."""
    await update.message.reply_text(
        "Welcome! Use /track_current to track from the current time or /track_purchase to track from the time of purchase."
    )

async def track_current(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Set tracking to start from the current time."""
    chat_id = update.effective_chat.id
    user_preferences[chat_id] = {"track_from": "current", "last_price": None}
    await update.message.reply_text("Tracking will start from the current time. You will receive notifications for price changes of 5% or more.")
    
    # Start tracking prices immediately
    await schedule_price_checks(update, context)

async def track_purchase(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Set tracking to start from the time of purchase."""
    chat_id = update.effective_chat.id
    user_preferences[chat_id] = {"track_from": "purchase", "last_price": None}
    await update.message.reply_text("Tracking will start from the time of purchase. You will receive notifications for price changes of 5% or more.")
    
    # Start tracking prices immediately
    await schedule_price_checks(update, context)

async def check_prices(context: ContextTypes.DEFAULT_TYPE):
    """Check prices and send notifications for significant profit changes."""
    global last_notified_price

    job = context.job
    chat_id = job.context  # Get the chat_id from the job context

    for pair in top_pairs:
        data = fetch_market_data(pair)
        if data.empty or len(data) < 2:
            continue

        # Get the recent close price
        recent_close = data["close"].iloc[-1]

        # Get user preference
        user_pref = user_preferences.get(chat_id)
        if user_pref is None or user_pref["track_from"] is None:
            continue  # Skip if user preference is not set

        if user_pref["track_from"] == "current":
            # Track from the current price
            previous_close = user_pref["last_price"] if user_pref["last_price"] is not None else recent_close
        else:
            # Track from the purchase price (initial price)
            previous_close = user_pref["last_price"] if user_pref["last_price"] is not None else recent_close

        # Calculate the percentage profit
        percentage_profit = ((recent_close - previous_close) / previous_close) * 100

        # Only send notification if the profit exceeds 5%
        if percentage_profit >= 5:
            user_preferences[chat_id]["last_price"] = recent_close  # Update the last tracked price
            message = (
                f"🚀 *{pair}* has increased by {percentage_profit:.2f}%!\n"
                f"💰 New Price: {recent_close:.2f} USDT"
            )
            await context.bot.send_message(
                chat_id=chat_id, text=message, parse_mode="Markdown"
            )

async def schedule_price_checks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Schedule price checks to run every minute."""
    if context.job_queue is None:
        logging.error("Job queue is not available.")
        return
    context.job_queue.run_repeating(
        check_prices, interval=60, chat_id=update.effective_chat.id
    )
    await update.message.reply_text(
        "Price monitoring started. You will receive notifications for significant price changes."
    )


# Main function
async def main():
    """Run the Telegram bot."""
    token = os.getenv("TELEGRAM_TOKEN")
    application = Application.builder().token(token).build()

    # Command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("track_current", track_current))
    application.add_handler(CommandHandler("track_purchase", track_purchase))

    # Run the bot
    logging.info("Bot is running...")
    await application.run_polling()


if __name__ == "__main__":
    # Allow nested event loops
    nest_asyncio.apply()

    # Run the main function
    asyncio.run(main())
