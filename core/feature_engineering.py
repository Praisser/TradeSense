import pandas as pd
import ta

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Add indicators
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['macd'] = ta.trend.MACD(df['close']).macd_diff()
    df['ema_fast'] = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
    df['ema_slow'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()

    # Target label: Good trade if price rises >2×ATR in next 5 candles
    df['future_close'] = df['close'].shift(-5)
    df['target'] = ((df['future_close'] - df['close']) > (2 * df['atr'])).astype(int)

    df.dropna(inplace=True)
    return df
