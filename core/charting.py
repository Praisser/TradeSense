import mplfinance as mpf
import pandas as pd
import os

def plot_chart_with_levels(df, sr_result=None, trend_info=None, save_path="latest_chart.png"):
    # Prepare OHLC data
    df = df.copy()
    df.set_index("time", inplace=True)

    # Convert to float (if needed)
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col].astype(float)

    # Build overlays: trendline + support/resistance
    apds = []

    if sr_result:
        for level in sr_result.get("support_levels", []):
            apds.append(mpf.make_addplot([level]*len(df), color='green'))
        for level in sr_result.get("resistance_levels", []):
            apds.append(mpf.make_addplot([level]*len(df), color='red'))

    if trend_info:
        slope = trend_info.get("slope", 0)
        trend_line = [df["close"].iloc[0] + slope * i for i in range(len(df))]
        apds.append(mpf.make_addplot(trend_line, color='blue'))

    # Save chart
    mpf.plot(df, type='candle', style='charles', addplot=apds,
             volume=False, savefig=dict(fname=save_path, dpi=100))

    return save_path
