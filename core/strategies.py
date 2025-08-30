import os
from core.mt5_utils import fetch_latest_data
from core.charting import plot_chart_with_levels


def ma_crossover_signal(symbol, timeframe, capital):
    df = fetch_latest_data(symbol, timeframe, bars=100)
    df['ema_fast'] = df['close'].ewm(span=5).mean()
    df['ema_slow'] = df['close'].ewm(span=12).mean()

    signal = int(df['ema_fast'].iloc[-1] > df['ema_slow'].iloc[-1])
    confidence = 0.9 if signal else 0.1
    atr = df['high'].rolling(14).max().iloc[-1] - df['low'].rolling(14).min().iloc[-1]

    chart_path = "temp/chart_ma.png"
    os.makedirs("temp", exist_ok=True)
    plot_chart_with_levels(df, sr_result=None, trend_info=None, save_path=chart_path)

    return {
        "confidence": confidence,
        "price": df["close"].iloc[-1],
        "atr": atr,
        "chart_path": chart_path
    }


def macd_signal_strategy(symbol, timeframe, capital):
    df = fetch_latest_data(symbol, timeframe, bars=100)
    df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()

    signal = int(df['macd'].iloc[-1] > 0)
    confidence = 0.85 if signal else 0.15
    atr = df['high'].rolling(14).max().iloc[-1] - df['low'].rolling(14).min().iloc[-1]

    chart_path = "temp/chart_macd.png"
    os.makedirs("temp", exist_ok=True)
    plot_chart_with_levels(df, sr_result=None, trend_info=None, save_path=chart_path)

    return {
        "confidence": confidence,
        "price": df["close"].iloc[-1],
        "atr": atr,
        "chart_path": chart_path
    }


def pattern_trigger_strategy(symbol, timeframe, capital):
    df = fetch_latest_data(symbol, timeframe, bars=100)
    signal = int(df['close'].iloc[-1] > df['open'].iloc[-1])
    confidence = 0.8 if signal else 0.2
    atr = df['high'].rolling(14).max().iloc[-1] - df['low'].rolling(14).min().iloc[-1]

    chart_path = "temp/chart_pattern.png"
    os.makedirs("temp", exist_ok=True)
    plot_chart_with_levels(df, sr_result=None, trend_info=None, save_path=chart_path)

    return {
        "confidence": confidence,
        "price": df["close"].iloc[-1],
        "atr": atr,
        "chart_path": chart_path
    }
