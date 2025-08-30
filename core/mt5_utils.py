import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta

def initialize_mt5():
    if not mt5.initialize():
        raise RuntimeError("MetaTrader 5 initialization failed")
    else:
        print("✅ Connected to MetaTrader 5")

def fetch_latest_data(symbol: str, timeframe: str, bars: int = 100) -> pd.DataFrame:
    """Fetch recent data for real-time prediction"""
    tf_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1
    }
    tf = tf_map.get(timeframe, mt5.TIMEFRAME_M15)
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
    if rates is None:
        raise ValueError("❌ Failed to fetch data from MT5.")
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def fetch_historical_data(symbol: str, timeframe: str, bars: int = 500) -> pd.DataFrame:
    """Fetch historical data for backtesting"""
    tf_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1
    }
    tf = tf_map.get(timeframe, mt5.TIMEFRAME_M15)
    start_time = datetime.now() - timedelta(minutes=bars * 2)
    rates = mt5.copy_rates_from(symbol, tf, start_time, bars)
    if rates is None:
        raise ValueError("❌ Failed to fetch historical MT5 data.")
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df
