import numpy as np
import pandas as pd


def compute_polynomial_bands(df, degree=4, window=60, k=2):
    df = df.copy()
    df['Poly_Center'] = np.nan
    df['Upper_Band'] = np.nan
    df['Lower_Band'] = np.nan

    closes = df['Close'].values

    for i in range(window, len(df)):
        # closing values up to i-1
        y = closes[i-window:i]
        # numbered x values from 0 - (window-1)
        x = np.arange(window)

        # fit polynomial based on degree
        coeffs = np.polyfit(x, y, degree)
        # create polynomial function based on coefficients
        poly = np.poly1d(coeffs)

        # calculate fitted values and error for each data point to find sigma
        fitted_window = poly(x)
        error = y - fitted_window

        sigma = np.sqrt(np.mean(error**2))

        # now calculate the center value at the window-th point
        center = poly(window)

        idx = df.index[i]
        df.at[idx, 'Poly_Center'] = center
        df.at[idx, 'Upper_Band'] = center + k * sigma
        df.at[idx, 'Lower_Band'] = center - k * sigma

    return df


def assign_signals(df):
    df = df.copy()
    df["Signal"] = np.nan

    # Exit when breakout above upper band
    df.loc[df["Close"] > df["Upper_Band"], "Signal"] = 0

    # Enter when breakdown below lower band
    df.loc[df["Close"] < df["Lower_Band"], "Signal"] = 1

    return df


def assign_positions(df):
    df = df.copy()

    df["Position"] = df["Signal"].ffill().fillna(0)
    df["Position"] = df["Position"].shift(1).fillna(0)

    return df


def apply_strategy(df, investment=10000):
    df = df.copy()
    df["Daily_Return"] = df["Close"].pct_change().fillna(0)
    df["Strategy_Return"] = df["Daily_Return"] * df["Position"]
    df["Strategy_Portfolio_Value"] = investment * (1 + df["Strategy_Return"]).cumprod()
    return df


def simple_strategy(df, investment=10000):
    df = df.copy()
    df["Simple_Portfolio_Value"] = investment * (1 + df["Daily_Return"]).cumprod()
    return df


def run_full_analysis(df, degree=4, window=60, k=2, investment=10000):

    df = df.copy()
    df = compute_polynomial_bands(df, degree=degree, window=window, k=k)
    df = assign_signals(df)
    df = assign_positions(df)
    df = apply_strategy(df, investment=investment)
    df = simple_strategy(df, investment=investment)
    return df


def calculate_metrics(df):
    metrics = {}

    # Final values
    metrics['strategy_final'] = df["Strategy_Portfolio_Value"].iloc[-1]
    metrics['simple_final'] = df["Simple_Portfolio_Value"].iloc[-1]

    # Total returns
    initial_investment = df["Strategy_Portfolio_Value"].iloc[0] if len(df) > 0 else 10000
    metrics['strategy_return_pct'] = ((metrics['strategy_final'] - initial_investment) / initial_investment) * 100
    metrics['simple_return_pct'] = ((metrics['simple_final'] - initial_investment) / initial_investment) * 100

    # Win rate (percentage of days with positive strategy returns when in position)
    in_position = df[df["Position"] == 1]
    if len(in_position) > 0:
        winning_days = len(in_position[in_position["Strategy_Return"] > 0])
        metrics['win_rate'] = (winning_days / len(in_position)) * 100
    else:
        metrics['win_rate'] = 0

    # Number of trades (complete entry-exit pairs)
    position_changes = df["Position"].diff()
    entries = (position_changes == 1).sum()  # Position changes from 0 to 1
    exits = (position_changes == -1).sum()   # Position changes from 1 to 0
    # Count complete trades (min of entries and exits)
    metrics['num_trades'] = min(entries, exits)

    return metrics


def calculate_yearly_returns(df):
    df = df.copy()
    df['Year'] = df.index.year

    yearly_returns = df.groupby('Year').agg({
        'Strategy_Portfolio_Value': lambda x: ((x.iloc[-1] / x.iloc[0]) - 1) * 100 if len(x) > 0 else 0,
        'Simple_Portfolio_Value': lambda x: ((x.iloc[-1] / x.iloc[0]) - 1) * 100 if len(x) > 0 else 0
    })

    yearly_returns.columns = ['Strategy Return (%)', 'Buy-and-Hold Return (%)']

    return yearly_returns
