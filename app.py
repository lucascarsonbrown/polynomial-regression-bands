"""
Streamlit Dashboard for Polynomial Regression Trading Strategy
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
# import matplotlib
from datetime import datetime
from utils import run_full_analysis, calculate_metrics, calculate_yearly_returns

# Page configuration
st.set_page_config(
    page_title="Polynomial Regresstion in Trading Systems",
    layout="wide",
    initial_sidebar_state="expanded"
)

# NASDAQ-100 tickers
NASDAQ100_TICKERS = [
    "AAPL", "ABNB", "ADBE", "ADI", "ADP", "ADSK", "AEP", "AMD", "AMAT", "AMGN",
    "AMZN", "APP", "ARM", "ASML", "AVGO", "AXON", "AZN", "BIIB", "BKNG", "BKR",
    "CCEP", "CDNS", "CDW", "CEG", "CHTR", "CMCSA", "COST", "CPRT", "CRWD", "CSCO",
    "CSGP", "CSX", "CTAS", "CTSH", "DASH", "DDOG", "DXCM", "EA", "EXC", "FANG",
    "FAST", "FTNT", "GEHC", "GFS", "GILD", "GOOG", "GOOGL", "HON", "IDXX", "INTC",
    "INTU", "ISRG", "KDP", "KHC", "KLAC", "LIN", "LRCX", "LULU", "MAR", "MCHP",
    "MDLZ", "MELI", "META", "MNST", "PEP", "PLTR", "PYPL", "QCOM", "REGN", "ROP",
    "ROST", "SBUX", "SHOP", "SNPS", "TEAM", "TMUS", "TSLA", "TTD", "TTWO", "TXN",
    "VRSK", "VRTX", "WBD", "WDAY", "XEL", "ZS"
]

DATA_DIRECTORY = "nasdaq100_data"


@st.cache_data
def load_stock_data(symbol):
    """Load stock data from CSV file."""
    # Get the directory where app.py is located
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, DATA_DIRECTORY, f"{symbol}.csv")

    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(
        csv_path,
        parse_dates=["Date"],
        index_col="Date",
        usecols=["Date", "Open", "Close"]
    )

    # Convert to proper DatetimeIndex and remove timezone
    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)

    df = df.sort_index()

    return df


def main():
    st.title("Polynomial Regresstion in Trading Systems")
    st.markdown("Explore polynomial regression-based trading strategies across NASDAQ-100 stocks.")

    # Sidebar - Parameters
    st.sidebar.header("Strategy Parameters")

    degree = st.sidebar.slider(
        "Polynomial Degree",
        min_value=1,
        max_value=10,
        value=4,
        help="Degree of the polynomial regression (higher = more flexible)"
    )

    window = st.sidebar.slider(
        "Window Size (days)",
        min_value=10,
        max_value=90,
        value=60,
        help="Number of days to use for regression calculation"
    )

    k_value = st.sidebar.slider(
        "Band Multiplier (k)",
        min_value=0.5,
        max_value=5.0,
        value=2.0,
        step=0.1,
        help="Standard deviation multiplier for band width"
    )

    investment = st.sidebar.number_input(
        "Initial Investment ($)",
        min_value=1000,
        max_value=1000000,
        value=10000,
        step=1000,
        help="Starting portfolio value"
    )

    st.sidebar.markdown("---")

    # Stock selector in sidebar
    st.sidebar.header("Stock Selection")

    # Initialize session state for selected stocks
    if 'selected_stocks' not in st.session_state:
        st.session_state.selected_stocks = []

    # Searchable multiselect dropdown
    selected = st.sidebar.multiselect(
        "Select stocks to analyze",
        options=NASDAQ100_TICKERS,
        default=st.session_state.selected_stocks,
        help="Search and select multiple stocks"
    )

    # Update session state
    st.session_state.selected_stocks = selected

    st.sidebar.markdown("---")

    # Date range selector in sidebar
    st.sidebar.header("Analysis Period")

    # Date constraints
    min_start_date = datetime(2010, 1, 1).date()
    max_end_date = datetime(2024, 10, 30).date()

    # Initialize date range in session state
    if 'start_date' not in st.session_state:
        st.session_state.start_date = datetime(2023, 1, 1).date()
    if 'end_date' not in st.session_state:
        st.session_state.end_date = max_end_date

    start_date = st.sidebar.date_input(
        "Start Date",
        value=st.session_state.start_date,
        min_value=min_start_date,
        max_value=max_end_date,
        help="Select the start date for analysis (min: 2010-01-01)"
    )

    end_date = st.sidebar.date_input(
        "End Date",
        value=st.session_state.end_date,
        min_value=min_start_date,
        max_value=max_end_date,
        help="Select the end date for analysis (max: 2024-10-30)"
    )

    st.session_state.start_date = start_date
    st.session_state.end_date = end_date

    st.sidebar.markdown("---")

    # Run Analysis Button in sidebar
    run_analysis = st.sidebar.button("🚀 Run Analysis", type="primary", use_container_width=True)

    # Analysis section
    if run_analysis and st.session_state.selected_stocks:
        with st.spinner("Running analysis..."):
            # Load and process data for all selected stocks
            results = {}

            for symbol in st.session_state.selected_stocks:
                df = load_stock_data(symbol)
                if df is not None:
                    # Run full analysis
                    processed_df = run_full_analysis(
                        df,
                        degree=degree,
                        window=window,
                        k=k_value,
                        investment=investment
                    )
                    results[symbol] = processed_df
                else:
                    st.warning(f"Could not load data for {symbol}")

            if not results:
                st.error("No valid stock data found. Please check your data directory.")
                return

            # Get min and max dates across all stocks
            all_dates = []
            for df in results.values():
                all_dates.extend(df.index.tolist())

            data_min_date = min(all_dates).date()
            data_max_date = max(all_dates).date()

            # Adjust dates if they're outside available data range
            adjusted_start = start_date
            adjusted_end = end_date

            if start_date < data_min_date:
                adjusted_start = data_min_date
                st.toast(f"Start date adjusted to {data_min_date} (earliest available data)", icon="ℹ️")

            if end_date > data_max_date:
                adjusted_end = data_max_date
                st.toast(f"End date adjusted to {data_max_date} (latest available data)", icon="ℹ️")

            if start_date > data_max_date or end_date < data_min_date:
                st.error(f"Selected date range is outside available data range ({data_min_date} to {data_max_date})")
                return

            # Filter data by date range and recalculate portfolio values
            filtered_results = {}
            for symbol, df in results.items():
                mask = (df.index.date >= adjusted_start) & (df.index.date <= adjusted_end)
                filtered_df = df.loc[mask].copy()

                # Recalculate portfolio values starting from the filtered start date
                if len(filtered_df) > 0:
                    # Recalculate daily returns
                    filtered_df["Daily_Return"] = filtered_df["Close"].pct_change().fillna(0)

                    # Recalculate strategy returns and portfolio value
                    filtered_df["Strategy_Return"] = filtered_df["Daily_Return"] * filtered_df["Position"]
                    filtered_df["Strategy_Portfolio_Value"] = investment * (1 + filtered_df["Strategy_Return"]).cumprod()

                    # Recalculate simple buy-and-hold portfolio value
                    filtered_df["Simple_Portfolio_Value"] = investment * (1 + filtered_df["Daily_Return"]).cumprod()

                filtered_results[symbol] = filtered_df

            st.markdown("---")

            # Chart 1: Polynomial Bands Visualization
            st.header("Polynomial Bands & Price Action")

            fig_bands = go.Figure()

            # Color palette for multiple stocks
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

            for idx, (symbol, df) in enumerate(filtered_results.items()):
                color = colors[idx % len(colors)]

                # Drop rows where polynomial bands are NaN (first 'window' days)
                df_valid = df.dropna(subset=['Poly_Center', 'Upper_Band', 'Lower_Band'])

                # Price line
                fig_bands.add_trace(go.Scatter(
                    x=df_valid.index,
                    y=df_valid['Close'],
                    mode='lines',
                    name=f'{symbol} Close',
                    line=dict(color=color, width=2)
                ))

                # Polynomial center
                fig_bands.add_trace(go.Scatter(
                    x=df_valid.index,
                    y=df_valid['Poly_Center'],
                    mode='lines',
                    name=f'{symbol} Poly Center',
                    line=dict(color=color, width=1, dash='dot'),
                    opacity=0.6
                ))

                # Upper band
                fig_bands.add_trace(go.Scatter(
                    x=df_valid.index,
                    y=df_valid['Upper_Band'],
                    mode='lines',
                    name=f'{symbol} Upper Band',
                    line=dict(color=color, width=1, dash='dash'),
                    opacity=0.4
                ))

                # Lower band
                fig_bands.add_trace(go.Scatter(
                    x=df_valid.index,
                    y=df_valid['Lower_Band'],
                    mode='lines',
                    name=f'{symbol} Lower Band',
                    line=dict(color=color, width=1, dash='dash'),
                    opacity=0.4,
                    fill='tonexty',
                    fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1)'
                ))

            fig_bands.update_layout(
                title=f"Price Action with Polynomial Bands (Degree={degree}, Window={window}, k={k_value})",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                hovermode='x unified',
                height=500,
                template='plotly_white'
            )

            st.plotly_chart(fig_bands, use_container_width=True)

            st.markdown("---")

            # Chart 2: Portfolio Value Comparison
            st.header("Portfolio Value Comparison")

            fig_portfolio = go.Figure()

            for idx, (symbol, df) in enumerate(filtered_results.items()):
                color = colors[idx % len(colors)]

                # Strategy portfolio
                fig_portfolio.add_trace(go.Scatter(
                    x=df.index,
                    y=df['Strategy_Portfolio_Value'],
                    mode='lines',
                    name=f'{symbol} Strategy',
                    line=dict(color=color, width=2)
                ))

                # Buy-and-hold portfolio
                fig_portfolio.add_trace(go.Scatter(
                    x=df.index,
                    y=df['Simple_Portfolio_Value'],
                    mode='lines',
                    name=f'{symbol} Buy-and-Hold',
                    line=dict(color=color, width=2, dash='dash'),
                    opacity=0.6
                ))

            fig_portfolio.update_layout(
                title=f"Portfolio Growth: Strategy vs Buy-and-Hold (Initial Investment: ${investment:,})",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                hovermode='x unified',
                height=500,
                template='plotly_white'
            )

            st.plotly_chart(fig_portfolio, use_container_width=True)

            st.markdown("---")

            # Metrics Section
            st.header("Performance Metrics")

            for symbol, df in filtered_results.items():
                with st.expander(f"{symbol} Metrics", expanded=len(filtered_results) == 1):
                    metrics = calculate_metrics(df)

                    # Display key metrics in columns
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric(
                            "Strategy Final Value",
                            f"${metrics['strategy_final']:,.2f}",
                            delta=f"{metrics['strategy_return_pct']:.2f}%"
                        )

                    with col2:
                        st.metric(
                            "Buy-and-Hold Final Value",
                            f"${metrics['simple_final']:,.2f}",
                            delta=f"{metrics['simple_return_pct']:.2f}%"
                        )

                    with col3:
                        st.metric(
                            "Win Rate",
                            f"{metrics['win_rate']:.2f}%"
                        )

                    with col4:
                        st.metric(
                            "Number of Trades",
                            f"{metrics['num_trades']}"
                        )

                    # Yearly returns table
                    st.markdown("**Yearly Returns Breakdown:**")
                    yearly_returns = calculate_yearly_returns(df)

                    if not yearly_returns.empty:
                        # Style the dataframe
                        try:
                            # Try to use background_gradient if matplotlib is available
                            st.dataframe(
                                yearly_returns.style.format("{:.2f}%").background_gradient(
                                    cmap='RdYlGn',
                                    axis=0,
                                    vmin=-50,
                                    vmax=50
                                ),
                                use_container_width=True
                            )
                        except ImportError:
                            # Fallback to simple formatting if matplotlib is not available
                            st.dataframe(
                                yearly_returns.style.format("{:.2f}%"),
                                use_container_width=True
                            )
                    else:
                        st.info("Not enough data for yearly breakdown")

    elif run_analysis and not st.session_state.selected_stocks:
        st.warning("⚠️ Please select at least one stock to analyze")

    # Footer
    st.markdown("---")
    st.markdown("""
    **About this dashboard:** This tool allows you to backtest a polynomial regression-based trading strategy
    on NASDAQ-100 stocks. The strategy uses polynomial bands (similar to Bollinger Bands) to generate
    entry and exit signals.

    - **Entry Signal:** When price breaks below the lower band
    - **Exit Signal:** When price breaks above the upper band
    """)


if __name__ == "__main__":
    main()
