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
    """Send a welcome message and start price checks."""
    chat_id = update.effective_chat.id
    await update.message.reply_text(
    "Welcome! I will notify you when the price of any top pair rises by more than 5%.\n\n"
    "Commands:\n"
    "/check - Get current prices and percentage changes every 5 minutes.\n"
    "/stop - Stop receiving periodic updates."
)

    # Schedule the periodic price check
    context.job_queue.run_repeating(
        check_prices, interval=300, first=0, chat_id=chat_id, name=str(chat_id)
    )

async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Stop the periodic price updates."""
    chat_id = update.effective_chat.id
    current_jobs = context.job_queue.get_jobs_by_name(str(chat_id))

    if not current_jobs:
        await update.message.reply_text("No active price checks to stop.")
        return

    for job in current_jobs:
        job.schedule_removal()

    await update.message.reply_text("Price checks have been stopped.")

async def check_prices(context: ContextTypes.DEFAULT_TYPE):
    """Check prices and send notifications for significant changes."""
    global last_notified_price

    job = context.job
    chat_id = job.chat_id

    for pair in top_pairs:
        data = fetch_market_data(pair)
        if data.empty or len(data) < 2:
            continue

        # Calculate percentage change
        recent_close = data["close"].iloc[-1]
        previous_close = data["close"].iloc[-2]
        percentage_change = ((recent_close - previous_close) / previous_close) * 100

        # Send update message for current price with percentage change
        update_message = (
            f"🔍 Current Price of *{pair}*: {recent_close:.2f} USDT\n"
            f"📈 Percentage Change: {percentage_change:.2f}%"
        )
        await context.bot.send_message(
            chat_id=chat_id, text=update_message, parse_mode="Markdown"
        )

        # Check if change exceeds 5% for notification
        if percentage_change > 5:
            if (
                pair not in last_notified_price
                or last_notified_price[pair] != recent_close
            ):
                last_notified_price[pair] = recent_close
                message = (
                    f"🚀 *{pair}* has increased by {percentage_change:.2f}%!\n"
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
        "Price monitoring started. Updates will be sent every minute."
    )


async def stop_price_checks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Stop all scheduled price checks."""
    context.job_queue.stop()
    await update.message.reply_text("Price monitoring stopped.")


# Main function
async def main():
    """Run the Telegram bot."""
    token = os.getenv("TELEGRAM_TOKEN")
    application = Application.builder().token(token).build()

    # Command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("stop", stop))
    application.add_handler(CommandHandler("start_monitoring", schedule_price_checks))
    application.add_handler(CommandHandler("stop_monitoring", stop_price_checks))

    # Run the bot
    logging.info("Bot is running...")
    await application.run_polling()


if __name__ == "__main__":
    # Allow nested event loops
    nest_asyncio.apply()

    # Run the main function
    asyncio.run(main())
