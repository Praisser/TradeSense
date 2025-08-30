from core.mt5_utils import fetch_historical_data
from core.feature_engineering import generate_features
from core.model_utils import load_model

def run_backtest(symbol, timeframe, candles, threshold, sl_mult, tp_mult, capital):
    df = fetch_historical_data(symbol, timeframe, candles + 50)
    df = generate_features(df)
    model = load_model()

    equity = [capital]
    wins = 0
    trades = 0

    for i in range(20, len(df)):
        row = df.iloc[i]
        features = row[['rsi', 'macd', 'ema_fast', 'ema_slow', 'atr']].values.reshape(1, -1)
        prob = model.predict_proba(features)[0, 1]
        if prob > threshold:
            price = row['close']
            atr = row['atr']
            sl = price - atr * sl_mult
            tp = price + atr * tp_mult

            future_high = df.iloc[i+1:i+4]['high'].max()
            future_low = df.iloc[i+1:i+4]['low'].min()

            trades += 1
            if future_high >= tp:
                wins += 1
                equity.append(equity[-1] + 10)
            elif future_low <= sl:
                equity.append(equity[-1] - 10)
            else:
                equity.append(equity[-1])

    winrate = (wins / trades * 100) if trades > 0 else 0
    pnl = equity[-1] - capital

    return {
        "equity_curve": equity,
        "winrate": winrate,
        "pnl": pnl
    }
