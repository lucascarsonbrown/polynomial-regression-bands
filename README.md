# Polynomial Regression Bands Trading Strategy

A comprehensive analysis and implementation of the Polynomial Regression Bands (PRB) trading strategy, inspired by Gilbert Raff's work and explored by financial analyst Steve Cohen. This project tests the strategy across 86 NASDAQ-100 stocks with over 3,500 parameter combinations to identify optimal settings.

## Visualizations

The project includes extensive visualizations:
- 3D surface plots showing parameter optimization landscapes
- Heatmaps comparing performance across parameter combinations
- Portfolio equity curves comparing strategy vs buy-and-hold
- Distribution of returns, Sharpe ratios, and drawdowns
- Win rate and profit factor analysis


### 1. Explore the Interactive App
Visit [https://prb-trading.streamlit.app/](https://prb-trading.streamlit.app/) to:
- Test different parameter combinations
- Visualize the strategy on any NASDAQ-100 stock
- Compare against buy-and-hold performance
- View detailed performance metrics

### 2. Read the Paper
Check out [Polynomial_Regression.pdf](./Polynomial_Regression.pdf) for:
- Mathematical derivation of the strategy
- Statistical analysis methodology
- Complete findings and interpretation
- Recommendations and limitations

## Technologies Used

- **Python**: Core implementation
- **NumPy/Pandas**: Data manipulation and analysis
- **Matplotlib**: Visualizations and charts
- **Streamlit**: Interactive web application
- **yfinance**: Historical stock data
- **SciPy**: Interpolation for 3D visualizations


Lucas Brown

MIT License