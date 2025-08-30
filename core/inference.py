from joblib import load
from core.feature_engineering import generate_features
from core.mt5_utils import fetch_latest_data
from core.charting import plot_chart_with_levels

model = load("models/forex_model.pkl")
features = ['rsi', 'macd', 'ema_fast', 'ema_slow', 'atr']

def predict_trade_signal(latest_df):
    df_feat = generate_features(latest_df)
    if df_feat.empty:
        return {"signal": 0, "confidence": 0.0}

    X_live = df_feat[features].iloc[[-1]]
    prob = model.predict_proba(X_live)[0][1]
    signal = int(prob > 0.5)
    return {'signal': signal, 'confidence': round(prob, 3)}

def predict_signal_from_mt5(symbol="EURUSD", timeframe="M15", capital=1000):
    df = fetch_latest_data(symbol, timeframe, bars=100)

    if df is None or df.empty:
        return {"signal": 0, "confidence": 0.0, "error": "No data"}

    result = predict_trade_signal(df)

    # Price Levels
    entry = df.iloc[-1]['close']
    r_level = df['high'].rolling(20).max().iloc[-1]
    s_level = df['low'].rolling(20).min().iloc[-1]

    # SL/TP logic using ATR
    atr = df["atr"].iloc[-1] if "atr" in df.columns else 0.0015
    sl = round(entry - 1.5 * atr, 5)
    tp = round(entry + 2.0 * atr, 5)

    # Chart for GUI
    chart_path = plot_chart_with_levels(df)

    result.update({
        "stop_loss": sl,
        "take_profit": tp,
        "chart_path": chart_path,
        "entry_price": round(entry, 5),
        "resistance": round(r_level, 5),
        "support": round(s_level, 5)
    })

    return result
