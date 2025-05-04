"""
HOW TO RUN THIS APPLICATION:
1. Ensure you have Python installed (3.7+ recommended)
2. Install required packages: 
   pip install streamlit pandas numpy yfinance scikit-learn plotly
3. Save this file as 'financial_ml_app.py'
4. Run the application:
   streamlit run financial_ml_app.py
5. The app will open in your default browser

Note: For Yahoo Finance data, you need an internet connection
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go  # Import for creating the graph
from datetime import date

# Page configuration
st.set_page_config(page_title="Financial ML App", layout="wide", page_icon="💹")

# Custom CSS for sidebar and table/chart styling
st.markdown("""
    <style>
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #d3eaf2;
        }

        section[data-testid="stSidebar"] input,
        section[data-testid="stSidebar"] textarea,
        [data-testid="stFileUploaderDropzone"] {
            background-color: white !important;
            color: black !important;
            border: none;
        }

        div.stButton > button {
            background-color: #5dade2 !important;
            color: white !important;
            border: none;
            border-radius: 8px;
            padding: 0.6em 1em;
            font-weight: bold;
        }

        div.stButton > button * {
            color: white !important;
        }

        div.stButton > button:hover {
            background-color: #3498db !important;
        }

        [data-testid="stFileUploader"] button {
            background-color: #5dade2 !important;
            color: white !important;
            font-weight: bold;
            border-radius: 5px;
            border: none;
        }

        [data-testid="stFileUploader"] button * {
            color: white !important;
        }

        [data-testid="stFileUploader"] small {
            display: block !important;
            color: gray;
            font-size: 0.8rem;
            margin-top: 5px;
        }

        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p {
            color: black !important;
        }

        /* Main area - only change table/chart background */
        .block-container .stDataFrame, 
        .block-container .stTable {
            background-color: white !important;
            color: black !important;
            border-radius: 10px;
            padding: 10px;
        }

        .block-container .js-plotly-plot .plotly {
            background-color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

# Welcome Section
# Create a three-column layout
left_col, center_col, right_col = st.columns([1, 2, 1])

with center_col:
    st.markdown("""
                <h1 style='text-align: left; color: white; margin-left: -360px;'>💹 Welcome to FinTrend – Your Smart Financial Forecasting Hub!</h1>
                """, unsafe_allow_html=True)
    st.image("assets/welcome.gif", width=500)
    st.write("This app allows you to upload financial data, fetch live stock prices, and apply Linear Regression on it.")

with right_col:
    st.markdown("### ℹ️ Instructions")
    st.success("""
    • Upload a Kaggle dataset or fetch live stock data using a ticker symbol.  
    • The app cleans and processes the data for machine learning.  
    • Linear Regression is applied to predict future closing prices.  
    • Visualize model predictions and performance metrics interactively.
    """)
    st.markdown("### 📂 Select Data from Kaggle")
    st.success("""
    You can download financial datasets directly from: https://www.kaggle.com/datasets?search=Stocks
    """)

# Sidebar 
st.sidebar.title("ℹ️ About")
st.sidebar.info(
    "FinTrend is a financial machine learning app that helps you explore financial data and apply ML models like Linear Regression. "
    "You can upload your own dataset or fetch live stock prices using Yahoo Finance. "
    "Get predictions and visualize them with interactive charts!"
)

# Sidebar - Dataset Upload + Yahoo Finance
st.sidebar.title("📊 Upload/Fetch Data")


uploaded_file = st.sidebar.file_uploader("Upload your Kaggle dataset (.csv)", type="csv")

ticker = st.sidebar.text_input("Enter Ticker Symbol (e.g., AAPL, MSFT):")
start_date = st.sidebar.date_input("Start Date", value=date(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", value=date.today())

if st.sidebar.button("📥 Fetch Yahoo Finance Data"):
    if ticker and start_date < end_date:
        try:
            df = yf.download(ticker, start=start_date, end=end_date)
            st.success(f"Successfully fetched {ticker} data!")
            st.write(df.tail())
        except Exception as e:
            st.error(f"Failed to fetch data: {e}")
    else:
        st.warning("Please enter a valid ticker and date range.")

# Load Uploaded CSV
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Kaggle CSV uploaded successfully!")
    st.write(df.head())

# Proceed only if data is available
if 'df' in locals():
    st.subheader("Preview of Data")
    st.write(df.head())

    # Prepare data for ML
    df = df.dropna()
    if 'Close' not in df.columns:
        st.error("'Close' column not found in dataset.")
    else:
        df['Target'] = df['Close'].shift(-1)
        df.dropna(inplace=True)

        X = df[['Close']]
        y = df['Target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Linear Regression
        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Display metrics
        st.subheader("📊 Model Evaluation")
        st.write("Mean Squared Error:", mean_squared_error(y_test, predictions))
        st.write("R² Score:", r2_score(y_test, predictions))

        # Plot actual vs predicted with regression line
        scatter = go.Scatter(x=y_test, y=predictions, mode='markers', name='Predicted')
        best_fit_line = go.Scatter(
            x=y_test,
            y=model.predict(y_test.values.reshape(-1, 1)),
            mode='lines',
            name='Best Fit Line',
            line=dict(color='blue', width=2)
        )
        perfect_line = go.Scatter(
            x=y_test,
            y=y_test,
            mode='lines',
            name='Perfect Line (y = x)',
            line=dict(color='red', dash='dash')
        )

        layout = go.Layout(
            title="Actual vs Predicted Closing Prices",
            xaxis_title='Actual',
            yaxis_title='Predicted',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black')
        )

        fig = go.Figure(data=[scatter, best_fit_line, perfect_line], layout=layout)
        st.plotly_chart(fig)

else:
    st.info("Upload a dataset or fetch data to begin.")

# Quote at the bottom center
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>🧠 Quote of the Day</h3>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; font-size: 18px;'>
<em>“It’s not whether you’re right or wrong, but how much money you make when you’re right and how much you lose when you’re wrong.”</em><br>
— <strong>George Soros</strong>
</div>
""", unsafe_allow_html=True)

# Credits and Copyrights
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center;'>Made with ❤️ by Ahmed Saleh Riaz</h5>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 14px;'>© 2025 All Rights Reserved</p>", unsafe_allow_html=True)