import streamlit as st
import matplotlib.pyplot as plt
from streamlit_autorefresh import st_autorefresh
import statsmodels.api as sm
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import json
import os
from datetime import datetime, timedelta
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="Custom Stock Index Tracker",
    layout="wide",
    initial_sidebar_state="expanded"
)

if not os.path.exists('saved_indices.json'):
    with open('saved_indices.json', 'w') as f:
        json.dump({}, f)

def load_saved_indices():
    with open('saved_indices.json', 'r') as f:
        return json.load(f)

def save_indices(indices):
    with open('saved_indices.json', 'w') as f:
        json.dump(indices, f, indent=4)

@st.cache_data(show_spinner=False)
def fetch_intraday_data(symbol, interval='5m', period='1d'):
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(interval=interval, period=period)
        if hist.empty:
            return None
        hist = hist.reset_index()
        return hist[['Datetime', 'Close']]
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None

@st.cache_data(show_spinner=False)
def fetch_daily_data(symbol, start_date=None, end_date=None):
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(start=start_date, end=end_date)
        if hist.empty:
            return None
        hist = hist.reset_index()
        return hist[['Date', 'Close']]
    except Exception as e:
        st.error(f"Error fetching daily data for {symbol}: {e}")
        return None

def calculate_returns(data, date_column='Datetime'):
    df = pd.DataFrame(data)
    df.set_index(date_column, inplace=True)
    returns = df['Close'].pct_change().dropna()
    return returns

def calculate_cumulative_returns(returns):
    cumulative_returns = (1 + returns).cumprod()
    return cumulative_returns

def normalize_weights(weights_dict):
    total_weight = sum(weights_dict.values())
    if total_weight == 0:
        return weights_dict
    return {k: v / total_weight for k, v in weights_dict.items()}

@st.cache_data(show_spinner=False)
def fetch_benchmark_intraday_data(symbol='^GSPC', interval='5m', period='1d'):
    return fetch_intraday_data(symbol, interval, period)

@st.cache_data(show_spinner=False)
def fetch_benchmark_daily_data(symbol='^GSPC', start_date=None, end_date=None):
    return fetch_daily_data(symbol, start_date, end_date)

def calculate_beta_and_residuals(index_returns, benchmark_returns):
    aligned_index, aligned_benchmark = index_returns.align(benchmark_returns, join='inner')
    X = aligned_benchmark.values.reshape(-1, 1)
    y = aligned_index.values.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, y)
    beta = model.coef_[0][0]
    alpha = model.intercept_[0]
    residuals = y.flatten() - (beta * X.flatten() + alpha)
    residuals_df = pd.DataFrame({'Residuals': residuals}, index=aligned_index.index)
    beta_adjusted_returns = pd.Series((beta * X.flatten() + alpha), index=aligned_index.index)
    return beta, alpha, residuals_df['Residuals'], beta_adjusted_returns

def find_continuous_regions(diff_mask, index):
    regions = []
    start = None
    for i in range(len(diff_mask)):
        if diff_mask.iloc[i] and start is None:
            start = index[i]
        elif not diff_mask.iloc[i] and start is not None:
            end = index[i - 1]
            regions.append((start, end))
            start = None
    if start is not None:
        regions.append((start, index[-1]))
    return regions

st_autorefresh(interval=3000000, key="data_refresh")
st.title("Stock Index Tracker")
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Go to", ["Summary", "Manage Indices", "View All Indices", "Historical Backtests"])

saved_indices = load_saved_indices()

if app_mode == "Manage Indices":
    st.header("Manage Your Indices")
    with st.expander("Add New Index"):
        index_name = st.text_input("Index Name", "")
        stock_symbols = st.text_input(
            "Enter stock symbols separated by commas (e.g., AAPL, MSFT, GOOGL)", "AAPL, MSFT, GOOGL"
        )
        stock_list = [symbol.strip().upper() for symbol in stock_symbols.split(",") if symbol.strip()]
        weights = {}
        st.subheader("Assign Weights")
        if stock_list:
            for symbol in stock_list:
                weight = st.slider(
                    f"Weight for {symbol}",
                    min_value=0.0,
                    max_value=1.0,
                    value=1.0 / len(stock_list),
                    step=0.01,
                )
                weights[symbol] = weight
            total_weight = sum(weights.values())
            if not np.isclose(total_weight, 1.0):
                st.warning(f"Weights sum to {total_weight:.2f}. They will be normalized.")
                weights = normalize_weights(weights)
        if st.button("Save Index"):
            if not index_name:
                st.error("Please provide a name for the index.")
            elif not stock_list:
                st.error("Please enter at least one stock symbol.")
            elif index_name in saved_indices:
                st.error(f"An index named '{index_name}' already exists. Please choose a different name.")
            else:
                saved_indices[index_name] = {
                    "stocks": weights
                }
                save_indices(saved_indices)
                st.success(f"Index '{index_name}' has been saved!")
    st.subheader("Existing Indices")
    if saved_indices:
        keys_to_delete = []
        for idx, (index_name, details) in enumerate(saved_indices.items()):
            with st.expander(f"{index_name}"):
                st.write("**Stocks and Weights:**")
                st.table(pd.DataFrame.from_dict(details['stocks'], orient='index', columns=['Weight']))
                if st.button(f"Delete '{index_name}'", key=f"delete_{index_name}"):
                    keys_to_delete.append(index_name)
        for key in keys_to_delete:
            del saved_indices[key]
            save_indices(saved_indices)
            st.success(f"Index '{key}' has been deleted!")
    else:
        st.info("No indices saved yet. Add a new index to get started.")

elif app_mode == "View All Indices":
    st.header("All Indices' Performance Analysis")
    if not saved_indices:
        st.info("No indices to display. Please add indices in the 'Manage Indices' section.")
    else:
        st.sidebar.subheader("Select Date Range for Daily Data")
        start_date = st.sidebar.date_input("Start Date", datetime(2024, 10, 1))
        end_date = st.sidebar.date_input("End Date", datetime.today())
        benchmark_symbol = '^GSPC'  # S&P 500

        # Fetch and cache benchmark data
        benchmark_intraday_data = fetch_benchmark_intraday_data(symbol=benchmark_symbol)
        if benchmark_intraday_data is not None:
            benchmark_intraday_returns = calculate_returns(benchmark_intraday_data, date_column='Datetime')
        else:
            st.error("Failed to fetch intraday benchmark data.")
            benchmark_intraday_returns = None

        benchmark_daily_data = fetch_benchmark_daily_data(symbol=benchmark_symbol, start_date=start_date, end_date=end_date)
        if benchmark_daily_data is not None:
            benchmark_daily_returns = calculate_returns(benchmark_daily_data, date_column='Date')
        else:
            st.error("Failed to fetch daily benchmark data.")
            benchmark_daily_returns = None

        for index_name, details in saved_indices.items():
            st.subheader(f"Index: {index_name}")
            weights = details['stocks']
            symbols = list(weights.keys())
            weight_values = list(weights.values())

            # Fetch intraday data for each stock
            intraday_data_frames = []
            for symbol in symbols:
                data = fetch_intraday_data(symbol)
                if data is not None:
                    intraday_data_frames.append(data.rename(columns={'Close': symbol}))
                else:
                    st.error(f"No intraday data found for {symbol}.")

            daily_data_frames = []
            for symbol in symbols:
                data = fetch_daily_data(symbol, start_date=start_date, end_date=end_date)
                if data is not None:
                    daily_data_frames.append(data.rename(columns={'Close': symbol}))
                else:
                    st.error(f"No daily data found for {symbol}.")

            # Intraday Analysis
            if intraday_data_frames:
                st.markdown("### Intraday Analysis")
                merged_intraday_df = intraday_data_frames[0]
                for df in intraday_data_frames[1:]:
                    merged_intraday_df = pd.merge(merged_intraday_df, df, on='Datetime', how='inner')
                if merged_intraday_df.empty:
                    st.warning(f"No overlapping intraday data for index '{index_name}'. Skipping intraday analysis.")
                else:
                    merged_intraday_df.set_index('Datetime', inplace=True)
                    index_returns = merged_intraday_df.pct_change().dropna()
                    index_returns_weighted = index_returns.multiply(pd.Series(weight_values, index=symbols)).sum(axis=1)
                    cumulative_returns = calculate_cumulative_returns(index_returns_weighted)
                    if benchmark_intraday_returns is not None:
                        aligned_index_returns, aligned_benchmark_returns = index_returns_weighted.align(benchmark_intraday_returns, join='inner')
                    else:
                        aligned_index_returns = index_returns_weighted
                        aligned_benchmark_returns = pd.Series([0]*len(index_returns_weighted), index=index_returns_weighted.index)
                    if benchmark_intraday_returns is not None:
                        beta, alpha, residuals, beta_adjusted_returns = calculate_beta_and_residuals(aligned_index_returns, aligned_benchmark_returns)
                    else:
                        beta, alpha, residuals, beta_adjusted_returns = 1.0, 0.0, pd.Series([0]*len(aligned_index_returns), index=aligned_index_returns.index), aligned_index_returns
                    beta_adjusted_residuals = residuals * beta
                    cumulative_residuals = (1 + beta_adjusted_residuals).cumprod()
                    ewm_beta_adj_residuals = beta_adjusted_residuals.ewm(span=20).mean()
                    ewm_index_returns = aligned_index_returns.ewm(span=20).mean()
                    ewm_benchmark_returns = aligned_benchmark_returns.ewm(span=20).mean() if benchmark_intraday_returns is not None else pd.Series()
                    st.markdown("#### Set Thresholds for Highlighting Differences (Intraday)")
                    threshold_intraday_graph1 = st.slider(
                        f"Threshold for Graph 1 (Index Returns vs S&P 500 Returns)",
                        min_value=0.0,
                        max_value=0.1,
                        value=0.01,
                        step=0.0001,
                        format="%.4f",
                        key=f"threshold_intraday_{index_name}_1"
                    )
                    col1, col2 = st.columns(2)
                    with col1:
                        fig1 = go.Figure()
                        fig1.add_trace(
                            go.Scatter(
                                x=aligned_index_returns.index,
                                y=aligned_index_returns.values,
                                mode='lines',
                                name=f"{index_name} Returns",
                                line=dict(width=2)
                            )
                        )
                        if benchmark_intraday_returns is not None:
                            fig1.add_trace(
                                go.Scatter(
                                    x=aligned_benchmark_returns.index,
                                    y=aligned_benchmark_returns.values,
                                    mode='lines',
                                    name='S&P 500 Returns',
                                    line=dict(width=2, dash='dash')
                                )
                            )
                        if benchmark_intraday_returns is not None:
                            diff = abs(aligned_index_returns - aligned_benchmark_returns)
                            diff_mask = diff > threshold_intraday_graph1
                            regions = find_continuous_regions(diff_mask, aligned_index_returns.index)
                            for start, end in regions:
                                fig1.add_vrect(
                                    x0=start,
                                    x1=end,
                                    fillcolor='red',
                                    opacity=0.2,
                                    layer='below',
                                    line_width=0,
                                )
                        fig1.update_layout(
                            title="Index Returns vs S&P 500 Returns (Intraday)",
                            xaxis_title="Datetime",
                            yaxis_title="Returns",
                            hovermode="x unified",
                            legend_title="Legend",
                            font=dict(size=12)
                        )
                        st.plotly_chart(fig1, use_container_width=True)
                    with col2:
                        fig2 = go.Figure()
                        fig2.add_trace(
                            go.Scatter(
                                x=beta_adjusted_residuals.index,
                                y=beta_adjusted_residuals.values,
                                mode='lines',
                                name='Beta-Adjusted Residuals',
                                line=dict(width=2)
                            )
                        )
                        fig2.update_layout(
                            title="Beta-Adjusted Residualized Returns (Intraday)",
                            xaxis_title="Datetime",
                            yaxis_title="Residual Returns",
                            hovermode="x unified",
                            legend_title="Legend",
                            font=dict(size=12)
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                    col3, col4 = st.columns(2)
                    with col3:
                        fig3 = go.Figure()
                        fig3.add_trace(
                            go.Scatter(
                                x=cumulative_residuals.index,
                                y=cumulative_residuals.values,
                                mode='lines',
                                name='Cumulative Residuals',
                                line=dict(width=2)
                            )
                        )
                        fig3.update_layout(
                            title="Cumulative Beta-Adjusted Residualized Returns (Intraday)",
                            xaxis_title="Datetime",
                            yaxis_title="Cumulative Residual Returns",
                            hovermode="x unified",
                            legend_title="Legend",
                            font=dict(size=12)
                        )
                        st.plotly_chart(fig3, use_container_width=True)
                    with col4:
                        fig4 = go.Figure()
                        fig4.add_trace(
                            go.Scatter(
                                x=ewm_beta_adj_residuals.index,
                                y=ewm_beta_adj_residuals.values,
                                mode='lines',
                                name='EWM Beta-Adjusted Residuals',
                                line=dict(width=2)
                            )
                        )
                        fig4.update_layout(
                            title="EWM Beta-Adjusted Residualized Returns (Intraday)",
                            xaxis_title="Datetime",
                            yaxis_title="EWM Residual Returns",
                            hovermode="x unified",
                            legend_title="Legend",
                            font=dict(size=12)
                        )
                        st.plotly_chart(fig4, use_container_width=True)
                    st.markdown("#### EWM of Returns vs S&P 500 (Intraday)")
                    fig5 = go.Figure()
                    fig5.add_trace(
                        go.Scatter(
                            x=ewm_index_returns.index,
                            y=ewm_index_returns.values,
                            mode='lines',
                            name=f"{index_name} EWM Returns",
                            line=dict(width=2)
                        )
                    )
                    if benchmark_intraday_returns is not None:
                        fig5.add_trace(
                            go.Scatter(
                                x=ewm_benchmark_returns.index,
                                y=ewm_benchmark_returns.values,
                                mode='lines',
                                name='S&P 500 EWM Returns',
                                line=dict(width=2, dash='dash')
                            )
                        )
                    fig5.update_layout(
                        title="Exponentially Weighted Rolling Mean of Returns (Intraday)",
                        xaxis_title="Datetime",
                        yaxis_title="EWM Returns",
                        hovermode="x unified",
                        legend_title="Legend",
                        font=dict(size=12)
                    )
                    st.plotly_chart(fig5, use_container_width=True)
            else:
                st.warning(f"No intraday data available for index '{index_name}'.")

            # Daily Analysis
            if daily_data_frames:
                st.markdown("### Daily Analysis (From October 1, 2024)")
                merged_daily_df = daily_data_frames[0]
                for df in daily_data_frames[1:]:
                    merged_daily_df = pd.merge(merged_daily_df, df, on='Date', how='inner')
                if merged_daily_df.empty:
                    st.warning(f"No overlapping daily data for index '{index_name}'. Skipping daily analysis.")
                else:
                    merged_daily_df.set_index('Date', inplace=True)
                    index_returns = merged_daily_df.pct_change().dropna()
                    index_returns_weighted = index_returns.multiply(pd.Series(weight_values, index=symbols)).sum(axis=1)
                    if benchmark_daily_returns is not None:
                        aligned_index_returns, aligned_benchmark_returns = index_returns_weighted.align(benchmark_daily_returns, join='inner')
                    else:
                        aligned_index_returns = index_returns_weighted
                        aligned_benchmark_returns = pd.Series([0]*len(index_returns_weighted), index=index_returns_weighted.index)
                    if benchmark_daily_returns is not None:
                        beta, alpha, residuals, beta_adjusted_returns = calculate_beta_and_residuals(aligned_index_returns, aligned_benchmark_returns)
                    else:
                        beta, alpha, residuals, beta_adjusted_returns = 1.0, 0.0, pd.Series([0]*len(aligned_index_returns), index=aligned_index_returns.index), aligned_index_returns
                    beta_adjusted_residuals = residuals * beta
                    cumulative_residuals = (1 + beta_adjusted_residuals).cumprod()
                    ewm_beta_adj_residuals = beta_adjusted_residuals.ewm(span=20).mean()
                    ewm_index_returns = aligned_index_returns.ewm(span=20).mean()
                    ewm_benchmark_returns = aligned_benchmark_returns.ewm(span=20).mean() if benchmark_daily_returns is not None else pd.Series()
                    st.markdown("#### Set Thresholds for Highlighting Differences (Daily)")
                    threshold_daily_graph1 = st.slider(
                        f"Threshold for Graph 1 (Index Returns vs S&P 500 Returns)",
                        min_value=0.0,
                        max_value=0.1,
                        value=0.01,
                        step=0.0001,
                        format="%.4f",
                        key=f"threshold_daily_{index_name}_1"
                    )
                    col1, col2 = st.columns(2)
                    with col1:
                        fig1 = go.Figure()
                        fig1.add_trace(
                            go.Scatter(
                                x=aligned_index_returns.index,
                                y=aligned_index_returns.values,
                                mode='lines',
                                name=f"{index_name} Returns",
                                line=dict(width=2)
                            )
                        )
                        if benchmark_daily_returns is not None:
                            fig1.add_trace(
                                go.Scatter(
                                    x=aligned_benchmark_returns.index,
                                    y=aligned_benchmark_returns.values,
                                    mode='lines',
                                    name='S&P 500 Returns',
                                    line=dict(width=2, dash='dash')
                                )
                            )
                        if benchmark_daily_returns is not None:
                            diff = abs(aligned_index_returns - aligned_benchmark_returns)
                            diff_mask = diff > threshold_daily_graph1
                            regions = find_continuous_regions(diff_mask, aligned_index_returns.index)
                            for start, end in regions:
                                fig1.add_vrect(
                                    x0=start,
                                    x1=end,
                                    fillcolor='red',
                                    opacity=0.2,
                                    layer='below',
                                    line_width=0,
                                )
                        fig1.update_layout(
                            title="Index Returns vs S&P 500 Returns (Daily)",
                            xaxis_title="Date",
                            yaxis_title="Returns",
                            hovermode="x unified",
                            legend_title="Legend",
                            font=dict(size=12)
                        )
                        st.plotly_chart(fig1, use_container_width=True)
                    with col2:
                        fig2 = go.Figure()
                        fig2.add_trace(
                            go.Scatter(
                                x=beta_adjusted_residuals.index,
                                y=beta_adjusted_residuals.values,
                                mode='lines',
                                name='Beta-Adjusted Residuals',
                                line=dict(width=2)
                            )
                        )
                        fig2.update_layout(
                            title="Beta-Adjusted Residualized Returns (Daily)",
                            xaxis_title="Date",
                            yaxis_title="Residual Returns",
                            hovermode="x unified",
                            legend_title="Legend",
                            font=dict(size=12)
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                    col3, col4 = st.columns(2)
                    with col3:
                        fig3 = go.Figure()
                        fig3.add_trace(
                            go.Scatter(
                                x=cumulative_residuals.index,
                                y=cumulative_residuals.values,
                                mode='lines',
                                name='Cumulative Residuals',
                                line=dict(width=2)
                            )
                        )
                        fig3.update_layout(
                            title="Cumulative Beta-Adjusted Residualized Returns (Daily)",
                            xaxis_title="Date",
                            yaxis_title="Cumulative Residual Returns",
                            hovermode="x unified",
                            legend_title="Legend",
                            font=dict(size=12)
                        )
                        st.plotly_chart(fig3, use_container_width=True)
                    with col4:
                        fig4 = go.Figure()
                        fig4.add_trace(
                            go.Scatter(
                                x=ewm_beta_adj_residuals.index,
                                y=ewm_beta_adj_residuals.values,
                                mode='lines',
                                name='EWM Beta-Adjusted Residuals',
                                line=dict(width=2)
                            )
                        )
                        fig4.update_layout(
                            title="EWM Beta-Adjusted Residualized Returns (Daily)",
                            xaxis_title="Date",
                            yaxis_title="EWM Residual Returns",
                            hovermode="x unified",
                            legend_title="Legend",
                            font=dict(size=12)
                        )
                        st.plotly_chart(fig4, use_container_width=True)
                    st.markdown("#### EWM of Returns vs S&P 500 (Daily)")
                    fig5 = go.Figure()
                    fig5.add_trace(
                        go.Scatter(
                            x=ewm_index_returns.index,
                            y=ewm_index_returns.values,
                            mode='lines',
                            name=f"{index_name} EWM Returns",
                            line=dict(width=2)
                        )
                    )
                    if benchmark_daily_returns is not None:
                        fig5.add_trace(
                            go.Scatter(
                                x=ewm_benchmark_returns.index,
                                y=ewm_benchmark_returns.values,
                                mode='lines',
                                name='S&P 500 EWM Returns',
                                line=dict(width=2, dash='dash')
                            )
                        )
                    fig5.update_layout(
                        title="Exponentially Weighted Rolling Mean of Returns (Daily)",
                        xaxis_title="Date",
                        yaxis_title="EWM Returns",
                        hovermode="x unified",
                        legend_title="Legend",
                        font=dict(size=12)
                    )
                    st.plotly_chart(fig5, use_container_width=True)
                    st.markdown("#### Download Daily Data as CSV")
                    download_df = pd.DataFrame({
                        'Index Returns': aligned_index_returns,
                        'Benchmark Returns': aligned_benchmark_returns,
                        'Beta-Adjusted Returns': beta_adjusted_returns,
                        'Residuals': residuals,
                        'Beta-Adjusted Residuals': beta_adjusted_residuals,
                        'EWM Beta-Adjusted Residuals': ewm_beta_adj_residuals,
                        'Cumulative Residuals': cumulative_residuals,
                        'EWM Index Returns': ewm_index_returns,
                        'EWM Benchmark Returns': ewm_benchmark_returns
                    })
                    csv = download_df.to_csv().encode('utf-8')
                    st.download_button(
                        label="Download Data as CSV",
                        data=csv,
                        file_name=f'{index_name}_daily_data.csv',
                        mime='text/csv',
                    )
            else:
                st.warning(f"No daily data available for index '{index_name}'.")

            st.markdown("---")

        with st.expander("View Individual Indices Data"):
            for index_name, details in saved_indices.items():
                st.subheader(f"Index: {index_name}")
                weights = details['stocks']
                symbols = list(weights.keys())
                weight_values = list(weights.values())
                data_frames = []
                for symbol in symbols:
                    data = fetch_daily_data(symbol, start_date=start_date, end_date=end_date)
                    if data is not None:
                        data_frames.append(data.rename(columns={'Close': symbol}))
                    else:
                        st.error(f"No data found for {symbol}.")
                if not data_frames:
                    st.warning(f"No data available for index '{index_name}'.")
                    continue
                merged_df = data_frames[0]
                for df in data_frames[1:]:
                    merged_df = pd.merge(merged_df, df, on='Date', how='inner')
                if merged_df.empty:
                    st.warning(f"No overlapping data for index '{index_name}'.")
                    continue
                merged_df.set_index('Date', inplace=True)
                index_returns = merged_df.pct_change().dropna()
                index_returns_weighted = index_returns.multiply(pd.Series(weight_values, index=symbols)).sum(axis=1)
                if benchmark_daily_returns is not None:
                    aligned_index_returns, aligned_benchmark_returns = index_returns_weighted.align(benchmark_daily_returns, join='inner')
                else:
                    aligned_index_returns = index_returns_weighted
                    aligned_benchmark_returns = pd.Series([0]*len(index_returns_weighted), index=index_returns_weighted.index)
                if benchmark_daily_returns is not None:
                    beta, alpha, residuals, beta_adjusted_returns = calculate_beta_and_residuals(aligned_index_returns, aligned_benchmark_returns)
                else:
                    beta, alpha, residuals, beta_adjusted_returns = 1.0, 0.0, pd.Series([0]*len(aligned_index_returns), index=aligned_index_returns.index), aligned_index_returns
                beta_adjusted_residuals = residuals * beta
                cumulative_residuals = (1 + beta_adjusted_residuals).cumprod()
                ewm_beta_adj_residuals = beta_adjusted_residuals.ewm(span=20).mean()
                detailed_df = pd.DataFrame({
                    'Residuals': residuals,
                    'Beta-Adjusted Residuals': beta_adjusted_residuals,
                    'Cumulative Residuals': cumulative_residuals,
                    'EWM Beta-Adjusted Residuals': ewm_beta_adj_residuals,
                })
                st.dataframe(detailed_df)

elif app_mode == "Summary":
    st.header("Summary of Indices")
    if not saved_indices:
        st.info("No indices to display. Please add indices in the 'Manage Indices' section.")
    else:
        st.sidebar.subheader("Select Date Range for Summary")
        start_date = st.sidebar.date_input("Start Date", datetime(2024, 10, 1))
        end_date = st.sidebar.date_input("End Date", datetime.today())
        benchmark_symbol = '^GSPC'  # S&P 500

        # Fetch and cache benchmark daily data
        benchmark_daily_data = fetch_benchmark_daily_data(symbol=benchmark_symbol, start_date=start_date, end_date=end_date)
        if benchmark_daily_data is not None:
            benchmark_daily_returns = calculate_returns(benchmark_daily_data, date_column='Date')
        else:
            st.error("Failed to fetch daily benchmark data.")
            benchmark_daily_returns = None

        for index_name, details in saved_indices.items():
            st.subheader(f"Index: {index_name}")
            weights = details['stocks']
            symbols = list(weights.keys())
            weight_values = list(weights.values())

            daily_data_frames = []
            for symbol in symbols:
                data = fetch_daily_data(symbol, start_date=start_date, end_date=end_date)
                if data is not None:
                    daily_data_frames.append(data.rename(columns={'Close': symbol}))
                else:
                    st.error(f"No daily data found for {symbol}.")

            if daily_data_frames:
                merged_daily_df = daily_data_frames[0]
                for df in daily_data_frames[1:]:
                    merged_daily_df = pd.merge(merged_daily_df, df, on='Date', how='inner')
                if merged_daily_df.empty:
                    st.warning(f"No overlapping daily data for index '{index_name}'. Skipping.")
                    continue
                else:
                    merged_daily_df.set_index('Date', inplace=True)
                    index_returns = merged_daily_df.pct_change().dropna()
                    index_returns_weighted = index_returns.multiply(pd.Series(weight_values, index=symbols)).sum(axis=1)
                    if benchmark_daily_returns is not None:
                        aligned_index_returns, aligned_benchmark_returns = index_returns_weighted.align(benchmark_daily_returns, join='inner')
                    else:
                        aligned_index_returns = index_returns_weighted
                        aligned_benchmark_returns = pd.Series([0]*len(index_returns_weighted), index=index_returns_weighted.index)
                    if benchmark_daily_returns is not None:
                        beta, alpha, residuals, _ = calculate_beta_and_residuals(aligned_index_returns, aligned_benchmark_returns)
                    else:
                        beta, alpha, residuals, _ = 1.0, 0.0, pd.Series([0]*len(aligned_index_returns), index=aligned_index_returns.index), aligned_index_returns
                    beta_adjusted_residuals = residuals * beta
                    cumulative_residuals = (1 + beta_adjusted_residuals).cumprod()
                    ewm_beta_adj_residuals = beta_adjusted_residuals.ewm(span=20).mean()
                    # Only include the two specified charts
                    cols = st.columns(2)
                    with cols[0]:
                        fig1 = go.Figure()
                        fig1.add_trace(
                            go.Scatter(
                                x=cumulative_residuals.index,
                                y=cumulative_residuals.values,
                                mode='lines',
                                name='Cumulative Beta-Adjusted Residuals',
                                line=dict(width=2)
                            )
                        )
                        fig1.update_layout(
                            title="Cumulative Beta-Adjusted Residualized Returns (Daily)",
                            xaxis_title="Date",
                            yaxis_title="Cumulative Beta-Adjusted Residualized Returns",
                            hovermode="x unified",
                            legend_title="Legend",
                            font=dict(size=12)
                        )
                        st.plotly_chart(fig1, use_container_width=True)
                    with cols[1]:
                        fig2 = go.Figure()
                        fig2.add_trace(
                            go.Scatter(
                                x=ewm_beta_adj_residuals.index,
                                y=ewm_beta_adj_residuals.values,
                                mode='lines',
                                name='EWM Beta-Adjusted Residuals',
                                line=dict(width=2)
                            )
                        )
                        fig2.update_layout(
                            title="Exponentially Weighted Mean of Beta-Adjusted Residuals (Daily)",
                            xaxis_title="Date",
                            yaxis_title="EWM Beta-Adjusted Residuals",
                            hovermode="x unified",
                            legend_title="Legend",
                            font=dict(size=12)
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                    st.markdown("---")
            else:
                st.warning(f"No daily data available for index '{index_name}'.")
elif app_mode == "Historical Backtests":
    import streamlit as st
    import pandas as pd
    import numpy as np
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    from datetime import datetime

    st.header("Historical Backtests")
    st.write("This page allows you to perform historical backtesting with beta hedging.")

    @st.cache_data
    def load_data():
        eq1 = pd.read_csv('eq1.csv', parse_dates=True)
        eq2 = pd.read_csv('eq2.csv', parse_dates=True)
        eq_prices = pd.concat([eq1, eq2], axis=0)
        eq_prices.set_index('as_of_date', inplace=True)
        eq_prices.index = pd.to_datetime(eq_prices.index)
        eq_prices['EMInotChina'] = eq_prices['USEQ:IEMG'] - 0.25 * eq_prices['USEQ:MCHI']
        return eq_prices

    eq_prices = load_data()

    @st.cache_data
    def compute_benchmark_returns(eq_prices, start_date, end_date, names):
        benchmark_prices = eq_prices.loc[start_date:end_date, names]
        benchmark_returns = benchmark_prices.pct_change().mean(axis=1)
        benchmark_returns.replace([np.inf, -np.inf], np.nan, inplace=True)
        benchmark_returns.dropna(inplace=True)
        return benchmark_returns

    @st.cache_data
    def compute_index_returns(eq_prices, symbols, weights, start_date, end_date):
        index_prices = eq_prices[symbols].loc[start_date:end_date]
        weight_series = pd.Series(weights, index=symbols)
        weighted_prices = index_prices.multiply(weight_series, axis=1)
        index_price_series = weighted_prices.sum(axis=1)
        index_returns = index_price_series.pct_change().dropna()
        index_returns.replace([np.inf, -np.inf], np.nan, inplace=True)
        index_returns.dropna(inplace=True)
        return index_returns

    @st.cache_data
    def compute_residuals(index_returns, benchmark_returns):
        aligned_index_returns, aligned_benchmark_returns = index_returns.align(benchmark_returns, join='inner')
        aligned_data = pd.concat([aligned_index_returns, aligned_benchmark_returns], axis=1).dropna()
        aligned_index_returns = aligned_data.iloc[:, 0]
        aligned_benchmark_returns = aligned_data.iloc[:, 1]
        aligned_benchmark_returns.replace([np.inf, -np.inf], np.nan, inplace=True)
        aligned_index_returns.replace([np.inf, -np.inf], np.nan, inplace=True)
        X = sm.add_constant(aligned_benchmark_returns)
        y = aligned_index_returns.reindex(X.index)
        if y.isna().any():
            X = X.loc[~y.isna()]
            y = y.dropna()
        model = sm.OLS(y, X).fit()
        residuals = model.resid
        return residuals, model, aligned_index_returns, aligned_benchmark_returns

    def enhanced_backtest_strategy(residuals, index_returns, benchmark_returns, model, d=10, k=0.01, exit_threshold=0.00, smoothing_span=30, initial_index_price=100):
        # Ensure indices are datetime
        if not np.issubdtype(residuals.index.dtype, np.datetime64):
            residuals.index = pd.to_datetime(residuals.index, format='%Y%m%d')
        
        # Sort data
        residuals = residuals.sort_index()
        index_returns = index_returns.sort_index()
        benchmark_returns = benchmark_returns.sort_index()
        
        # Combine data
        combined_data = pd.concat([residuals, index_returns, benchmark_returns], axis=1, join='inner')
        combined_data.columns = ['Residuals', 'Index_Returns', 'Benchmark_Returns']
        
        # Re-assign variables
        residuals = combined_data['Residuals']
        index_returns = combined_data['Index_Returns']
        benchmark_returns = combined_data['Benchmark_Returns']
        
        # Correct cumulative residuals calculation
        cumulative_residuals = residuals.cumsum()
        
        # Apply smoothing
        smoothed = cumulative_residuals.ewm(span=smoothing_span, adjust=False).mean()
        
        # Compute jumps
        jump = smoothed.diff(d).fillna(0)
        
        # Generate positions based on jumps
        positions = []
        holding = False
        current_position = 0
        for jump_value in jump:
            if holding:
                if current_position == 1 and jump_value < exit_threshold:
                    positions.append(0)
                    holding = False
                    current_position = 0
                elif current_position == -1 and jump_value > -exit_threshold:
                    positions.append(0)
                    holding = False
                    current_position = 0
                else:
                    positions.append(current_position)
            else:
                if jump_value > k:
                    positions.append(1)
                    holding = True
                    current_position = 1
                elif jump_value < -k:
                    positions.append(-1)
                    holding = True
                    current_position = -1
                else:
                    positions.append(0)
        position = pd.Series(positions, index=jump.index)
        position_shifted = position.shift(1).fillna(0)
        
        # Get beta from the model
        beta = model.params[1]
        st.write(f"Beta is {beta}")
        
        # Compute hedged returns
        hedged_return = index_returns - beta * benchmark_returns
        
        # Compute strategy returns with beta hedging
        strategy_returns = position_shifted * hedged_return
        
        # Correct cumulative strategy returns calculation
        cumulative_strategy_returns = (1 + strategy_returns).cumprod() - 1
        
        # Compute index price series
        index_price_series = initial_index_price * (1 + index_returns).cumprod()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        ax1.plot(cumulative_strategy_returns.index, cumulative_strategy_returns.values, label='Strategy Cumulative Returns', color='blue')
        ax1.set_ylabel('Cumulative Returns')
        ax1.set_title('Strategy Cumulative Returns vs. Index Price')
        ax1.legend(loc='upper left')
        ax1.grid(True)
        ax1b = ax1.twinx()
        ax1b.plot(index_price_series.index, index_price_series.values, label='Index Price', color='green', alpha=0.6)
        ax1b.set_ylabel('Index Price')
        ax1b.legend(loc='upper right')
        ax2.plot(smoothed.index, smoothed.values, label='Smoothed Cumulative Residuals', color='gray')
        ax2.set_ylabel('Smoothed Cumulative Residuals')
        ax2.set_title('Smoothed Jumps and Strategy Positions')
        ax2.legend(loc='upper left')
        ax2.grid(True)
        ax2b = ax2.twinx()
        ax2b.plot(position_shifted.index, position_shifted.values, label='Position', color='red', alpha=0.3)
        ax2b.set_ylabel('Position')
        ax2b.set_ylim(-1.5, 1.5)
        ax2b.legend(loc='upper right')
        plt.xlabel('Date')
        plt.tight_layout()
        total_return = cumulative_strategy_returns.iloc[-1]
        num_days = (cumulative_strategy_returns.index[-1] - cumulative_strategy_returns.index[0]).days
        if num_days > 0:
            annualized_return = (1 + total_return) ** (365.25 / num_days) - 1
        else:
            annualized_return = np.nan
        
        return fig, total_return, annualized_return

    # Assuming saved_indices is defined elsewhere in your code
    if saved_indices:
        st.subheader("Backtesting Saved Indices")
        st.sidebar.subheader("Select Date Range for Backtest")
        start_date = st.sidebar.date_input("Start Date", datetime(2020, 1, 1))
        end_date = st.sidebar.date_input("End Date", datetime.today())
        start_date = pd.to_datetime(start_date).tz_localize(None)
        end_date = pd.to_datetime(end_date).tz_localize(None)
        start_date1 = pd.to_datetime('2020-01-02').tz_localize(None)
        end_date1 = pd.to_datetime('2024-11-10').tz_localize(None)
        available_tickers = eq_prices.columns.tolist()
        names = st.multiselect(
            "Select tickers for calculating beta with respect to",
            options=available_tickers,
            default=['USEQ:SPY']
        )
        names = [col for col in names if col in eq_prices.columns]


        benchmark_returns = compute_benchmark_returns(eq_prices, start_date, end_date, names)
        benchmark_returns1 = compute_benchmark_returns(eq_prices, start_date1, end_date1, names)
        if benchmark_returns.isna().all():
            st.error("Benchmark returns are all NaN. Check your data.")
            st.stop()
        if benchmark_returns1.isna().all():
            st.error("Benchmark returns are all NaN. Check your data.")
            st.stop()
        for index_name, details in saved_indices.items():
            with st.expander(f"{index_name}"):
                st.subheader(f"Backtest for Index: {index_name}")
                weights = details['stocks']
                symbols = list(weights.keys())
                weight_values = list(weights.values())
                symbols_with_prefix = ['USEQ:' + symbol for symbol in symbols]
                missing_symbols = [symbol for symbol in symbols_with_prefix if symbol not in eq_prices.columns]
                if missing_symbols:
                    st.warning(f"Symbols {missing_symbols} not found in the price data. They will be excluded from the index '{index_name}'.")
                available_symbols = [symbol for symbol in symbols_with_prefix if symbol in eq_prices.columns]
                if not available_symbols:
                    st.error(f"No available symbols for index '{index_name}'. Skipping.")
                    continue
                available_weights = [weights[symbol[5:]] for symbol in available_symbols]
                total_weight = sum(available_weights)
                if total_weight == 0:
                    st.error(f"Total weight is zero for index '{index_name}'. Skipping.")
                    continue
                adjusted_weights = [w / total_weight for w in available_weights]
                index_returns = compute_index_returns(eq_prices, available_symbols, adjusted_weights, start_date, end_date)
                index_returns1 = compute_index_returns(eq_prices, available_symbols, adjusted_weights, start_date1, end_date1)
                if index_returns.empty:
                    st.warning(f"No data available for index '{index_name}' in the selected date range.")
                    continue
                if index_returns1.empty:
                    st.warning(f"No data available for index '{index_name}' in the selected date range.")
                    continue
                residuals, model, aligned_index_returns, aligned_benchmark_returns = compute_residuals(index_returns, benchmark_returns)
                residuals1, model1, aligned_index_returns1, aligned_benchmark_returns1 = compute_residuals(index_returns1, benchmark_returns1)
                st.subheader("Backtest Parameters")
                d = st.number_input(f"Number of days to look back for detecting jumps (d) - {index_name}", min_value=1, max_value=100, value=10, key=f"d_{index_name}")
                k = st.number_input(f"Threshold magnitude for entry signals (k) - {index_name}", min_value=0.0, max_value=10.0, value=0.01, key=f"k_{index_name}")
                exit_threshold = st.number_input(f"Threshold magnitude for exit signals - {index_name}", min_value=0.0, max_value=10.0, value=0.00, key=f"exit_{index_name}")
                smoothing_span = st.number_input(f"Span parameter for exponential smoothing - {index_name}", min_value=1, max_value=100, value=30, key=f"smoothing_{index_name}")
                initial_index_price = st.number_input(f"Starting price of the index - {index_name}", min_value=0.0, value=100.0, key=f"initial_price_{index_name}")
                fig, total_return, annualized_return = enhanced_backtest_strategy(
                    residuals,
                    aligned_index_returns,
                    aligned_benchmark_returns,
                    model1,
                    d=int(d),
                    k=float(k),
                    exit_threshold=float(exit_threshold),
                    smoothing_span=int(smoothing_span),
                    initial_index_price=float(initial_index_price)
                )
                st.pyplot(fig)
                plt.close(fig)

                st.write(f"**Total Strategy Return:** {total_return:.2%}")
                st.write(f"**Approximate Annualized Strategy Return:** {annualized_return:.2%}")
    else:
        st.info("No indices saved yet. Add a new index to get started.")

    # Backtesting Custom Stock Selection
    st.subheader("Backtesting Custom Stock Selection")
    available_tickers = eq_prices.columns.tolist()
    subset_columns = st.multiselect(
        "Select stocks to include in the backtest:",
        options=available_tickers,
        default=available_tickers[:5]
    )
    selected_columns = [col for col in subset_columns if col in eq_prices.columns]
    if not selected_columns:
        st.warning("No valid stocks selected. Please select at least one stock.")
        st.stop()
    start_date = st.sidebar.date_input("Start Date", datetime(2020, 1, 1), key="custom_start_date")
    end_date = st.sidebar.date_input("End Date", datetime.today(), key="custom_end_date")
    start_date = pd.to_datetime(start_date).tz_localize(None)
    end_date = pd.to_datetime(end_date).tz_localize(None)
    available_tickers = eq_prices.columns.tolist()
    # names = st.multiselect(
    #     "Select tickers for calculating beta with respect to",
    #     options=available_tickers,
    #     default=['USEQ:SPY']
    # )
    # names = [col for col in names if col in eq_prices.columns]
    benchmark_returns = compute_benchmark_returns(eq_prices, start_date, end_date, names)
    if benchmark_returns.isna().all():
        st.error("Benchmark returns are all NaN. Check your data.")
        st.stop()
    index_prices = eq_prices[selected_columns].loc[start_date:end_date]
    index_returns = index_prices.pct_change().mean(axis=1)
    index_returns.replace([np.inf, -np.inf], np.nan, inplace=True)
    index_returns.dropna(inplace=True)
    if index_returns.empty:
        st.warning("No data available for the selected stocks in the selected date range.")
        st.stop()
    residuals, model, aligned_index_returns, aligned_benchmark_returns = compute_residuals(index_returns, benchmark_returns)
    st.subheader("OLS Regression Summary")
    st.text(model.summary())
    st.subheader("Backtest Parameters")
    d = st.number_input("Number of days to look back for detecting jumps (d)", min_value=1, max_value=100, value=10, key="custom_d")
    k = st.number_input("Threshold magnitude for entry signals (k)", min_value=0.0, max_value=10.0, value=0.01, key="custom_k")
    exit_threshold = st.number_input("Threshold magnitude for exit signals", min_value=0.0, max_value=10.0, value=0.00, key="custom_exit")
    smoothing_span = st.number_input("Span parameter for exponential smoothing", min_value=1, max_value=100, value=30, key="custom_smoothing")
    
    initial_index_price = st.number_input("Starting price of the index", min_value=0.0, value=100.0, key="custom_initial_price")
    fig, total_return, annualized_return = enhanced_backtest_strategy(
        residuals,
        aligned_index_returns,
        aligned_benchmark_returns,
        model,
        d=int(d),
        k=float(k),
        exit_threshold=float(exit_threshold),
        smoothing_span=int(smoothing_span),
        initial_index_price=float(initial_index_price)
    )
    st.pyplot(fig)
    plt.close(fig)
    st.write(f"**Total Strategy Return:** {total_return:.2%}")
    st.write(f"**Approximate Annualized Strategy Return:** {annualized_return:.2%}")
