# run_realtime_bot.py

from core.mt5_utils import initialize_mt5, fetch_latest_data
from core.inference import predict_trade_signal
import time

SYMBOL = "EURUSD"
TIMEFRAME = 15  # minutes
BARS = 100

initialize_mt5()

print(f"📡 Running real-time signal bot for {SYMBOL} every {TIMEFRAME} minutes...\n")

try:
    while True:
        df = fetch_latest_data(SYMBOL, bars=BARS)
        result = predict_trade_signal(df)

        if result['signal'] == 1:
            print(f"✅ GOOD TRADE SIGNAL | Confidence: {result['confidence']}")
        else:
            print(f"❌ No Trade | Confidence: {result['confidence']}")

        time.sleep(TIMEFRAME * 60)  # Wait until next candle
except KeyboardInterrupt:
    print("🛑 Bot stopped by user.")
