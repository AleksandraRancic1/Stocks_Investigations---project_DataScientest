import streamlit as st
import requests
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import load_model
import webbrowser
import plotly.express as px

# API URL
API_URL = "http://localhost:8000/docs"

# Title of the app
st.title("Stock Prediction API")

# Home page
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Introduction", "Modelling", "API Specifications", "Go to API", "Tests", "CI/CD Pipeline", "Further Improvements"])

if page == "Home":
    st.write("**Welcome to the Stock Prediction API**")
    st.write("Use the sidebar to navigate")

    st.image('/Users/aleksandra.rancic/Desktop/Home_picture.png')

elif page == "Introduction":
    st.subheader("Stock Prediction - Importance")
    st.write("""
        Stock prediction is crucial for investors and financial analysts as it helps in making informed 
        decisions regarding stock purchases and sales. Accurate predictions can lead to significant profits 
        and help in risk management by minimizing potential losses.
    """)

    st.subheader("Visualization of Stocks")
    st.write("""
        Visualizing stock data is essential for understanding market trends and patterns, enabling investors to make data-driven 
        decisions.
    """)

    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL for Apple):", "AAPL")
    start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("today"))

    if st.button("Get Stock Data"):
        if ticker:
            try:
                stock_data = yf.download(ticker, start=start_date, end=end_date)
                plt.figure(figsize=(10, 5))
                plt.plot(stock_data['Adj Close'], label='Adjusted Close Price', color='blue')
                plt.title(f'Adjusted Close Price of {ticker} from {start_date} to {end_date}')
                plt.xlabel('Date')
                plt.ylabel('Adjusted Close Price')
                plt.xticks(rotation=45)
                plt.legend()
                plt.grid()
                st.pyplot(plt)
            except Exception as e:
                st.error(f"An error occurred: {e}")

    st.subheader("Stock API - Business")
    st.write("""
        A stock prediction API enables businesses to integrate predictive analytics into their 
        applications. It provides users with real-time predictions, allowing them to optimize 
        their trading strategies, enhance customer experience, and ultimately drive revenue growth. 
        By leveraging advanced machine learning models, businesses can stay competitive in the fast-paced 
        financial markets.
    """)

    st.write("""
        ### Dataset Variables Explanation:
        
        The dataset contains historical stock prices with the following key variables:

        1. **Date**: The date of the stock data. This serves as the index and helps track stock price movements over time.
       
        2. **Open**: The price of the stock at the beginning of the trading day. It represents the first trade price when the market opens.
       
        3. **High**: The highest price reached by the stock during the trading day. This gives an idea of the stock's daily volatility.
       
        4. **Low**: The lowest price reached by the stock during the trading day. It indicates the minimum value investors were willing to pay for the stock.
       
        5. **Close**: The price of the stock at the end of the trading day. It is the most common value used in analysis and forecasting models.
       
        6. **Adj Close**: The adjusted closing price takes into account corporate actions such as dividends, stock splits, and new share issuance. It provides a more accurate representation of stock value over time.
       
        7. **Volume**: The number of shares traded during the day. It gives insights into the liquidity and overall market interest in the stock.
    """)

elif page == "Modelling":
    st.subheader("Data Preprocessing and Model Training")

    st.write("""
        The data preprocessing involved several key steps to prepare the stock price data for training the LSTM model:
        
        1. **Splitting the Data**: The dataset was divided into training and testing sets, with 80% of the data used for training and 20% reserved for testing.
        
        2. **Scaling**: The stock price data was normalized using MinMaxScaler to scale the values between 0 and 1. This is essential for LSTMs as they perform better on normalized data.
        
        3. **Creating Sequences**: Input sequences of 60 time steps were created for the LSTM model. Each input sequence helps the model learn the temporal dependencies in the data, which is crucial for predicting future stock prices.
    """)

    st.write("""
        The model used for this task is a Long Short-Term Memory (LSTM) network, which is a type of recurrent neural network (RNN) specifically designed to capture long-term dependencies in sequential data.
        LSTMs are particularly well-suited for time series forecasting due to their ability to retain information over long sequences, making them effective for understanding patterns and trends in stock prices.
    """)

    lstm_model = load_model('/Users/aleksandra.rancic/Desktop/MLOps_Project/MLOps/ML_Model_API/venv/lstm_model.h5')

    st.subheader("LSTM Model Summary")
    lstm_model.summary(print_fn=lambda x: st.text(x))

    st.subheader("Predictions and Metrics of the trained LSTM model")

    st.image('/Users/aleksandra.rancic/Desktop/TP_values.png', caption='Predicted vs Actual Stock Prices', use_column_width=True)
    st.image('/Users/aleksandra.rancic/Desktop/Metrics.png', caption='LSTM model metrics', use_column_width=True)

elif page == "API Specifications":
    st.title("Stock Prediction API Specifications")

    st.write("""
        The Stock Prediction API provides several endpoints for stock data processing, prediction, and monitoring model performance.
        It leverages LSTM models to predict future stock prices based on historical data and provides Prometheus metrics for monitoring. 
        Below are the available endpoints:
    """)

    st.subheader("/ (Root)")
    st.write("""
        **Description**: The root endpoint confirms that the API is running.
        **Method**: `GET`
    """)

    st.subheader("/preprocess")
    st.write("""
        **Description**: Preprocesses the given stock data (adjusted close price) by scaling it using MinMaxScaler.
        **Method**: `POST`
    """)

    st.subheader("/predict")
    st.write("""
        **Description**: Predicts the future stock price based on the given adjusted close price.
        **Method**: `POST`
    """)

    st.subheader("/download_stock_data")
    st.write("""
        **Description**: Downloads stock data for the given ticker symbol for the last 10 years and saves it as a CSV file.
        **Method**: `GET`
    """)

    st.subheader("/upload_and_predict")
    st.write("""
        **Description**: Uploads a CSV file with stock data and predicts future stock prices based on the adjusted close prices.
        **Method**: `POST`
    """)

    st.subheader("/metrics")
    st.write("""
        **Description**: Returns the precomputed evaluation metrics (MSE, MAE, etc.) of the LSTM model.
        **Method**: `GET`
	
	```
	Mean Squared Error (MSE), 
        Root Mean Squared Error (RMSE), 
        Mean Absolute Error (MAE), 
        Mean Absolute Percentage Error (MAPE), 
        R-squared (R²)
	```
    """)

    st.subheader("/evaluate")
    st.write("""
        **Description**: Evaluates the current LSTM model on the test data and checks if it meets the success criteria.
        **Method**: `GET`
    """)

    st.subheader("/retrain")
    st.write("""
        **Description**: Retrains the LSTM model using the training data and saves the updated model.
        **Method**: `POST`
    """)

elif page == "Go to API":
    webbrowser.open(API_URL)

elif page == "Tests":
    st.title("Testing the API")
    st.subheader("Manual Testing Locally")
    st.write("Here, we test the FastAPI endpoints manually using pytest and FastAPI's TestClient.")
    st.write("""
    The test cases include:
    - Testing the root endpoint
    - Downloading stock data
    - Preprocessing stock data
    - Evaluating the model
    - Making predictions
    - Retraining the model
    - Checking model metrics
    """)
    st.write("Below is an example of the `test_root` function:")

    st.code('''def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to API for Stock predictions. The API is running!"}''', language = "python")

    st.image('/Users/aleksandra.rancic/Desktop/Test_manual.png')

    st.subheader("Automating the process locally - Setting Up Cron Jobs for API Testing")
    st.write("""
    To automate API testing, you can set up a cron job that periodically runs your test suite. 
    The following example runs `pytest` every hour, logging the results to a file:
    """)
    st.code('''0 * * * * /path/to/venv/bin/pytest /path/to/test_api.py >> /path/to/test_logs.txt 2>&1''', language="bash")
    st.write("You can adjust the schedule and output directory as needed.")
    st.write("It is also possible to set up a Cron job to follow model performance locally.")
    
    def load_metrics_data(json_file):
        with open(json_file, 'r') as file:
            data = json.load(file)
        return pd.DataFrame(data)

    def convert_timestamp(df):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    st.subheader("Model Performance Over Time")
    json_file_path = '/Users/aleksandra.rancic/Desktop/MLOps_Project/MLOps/ML_Model_API/venv/model_performance_log.json'
    if json_file_path:

        try:
            df = load_metrics_data(json_file_path)
            df = convert_timestamp(df)
            st.dataframe(df)
        
            fig = px.line(df, x = 'timestamp', y=['test_loss', 'test_mae', 'test_mse'], title='Model Evaluation Metrics Over Time', labels = {'value': 'Metric Value', 'timestamp': 'Timestamp'}, markers = True)
            fig.update_layout(xaxis_title='Timestamp', yaxis_title='Metrics', legend_title_text='Metrics', template='plotly_dark')
            st.plotly_chart(fig)
        except Exception as e:
            st.error(f"An error occured: {e}")

    st.subheader("Dockerizing the API with Cron Jobs")
    st.write("""
    To test the API in a production-like environment, you can containerize your FastAPI app and schedule automated tests using cron within the Docker container.
    Here is an example 'Dockerfile' and 'docker-compose.yml' setup.
    """)

    st.code('''FROM python:3.10-slim
    WORKDIR /app
    COPY requirements.txt .
    RUN pip install -r requirements.txt
    COPY . .
    CMD ["uvicorn", "stock_prediction_api:api", "--host", "0.0.0.0", "--port", "8000"]''', language="dockerfile")
    st.write("And here is the `docker-compose.yml`:")
    st.code('''version: '3.9'
    services:
      web:
        build: .
        ports:
          - "8000:8000"
        volumes:
          - .:/app
        command: ["uvicorn", "stock_prediction_api:api", "--host", "0.0.0.0", "--port", "8000"]

      cron:
        build: .
        command: crond -f
        volumes:
          - .:/app''', language="yaml")
    st.image("/Users/aleksandra.rancic/Desktop/Docker.png")

elif page == "CI/CD Pipeline":
    st.title("CI/CD Pipeline")
    st.write("""
        ### Continuous Integration and Continuous Deployment (CI/CD) Pipeline Overview:

        The CI/CD pipeline for the Stock Prediction API ensures that changes to the codebase 
        are automatically tested, built, and deployed to the production environment.
        
        - **Continuous Integration (CI)**: Ensures that code changes are regularly integrated and tested using automated unit tests and integration tests.
        - **Continuous Deployment (CD)**: Automatically deploys the latest version of the API to the production environment once the tests are passed.
        
        Key steps in the CI/CD pipeline:
        1. **Code Push**: Developers push the latest changes to the version control system (GitHub).
        2. **Automated Testing**: The pipeline triggers automated tests using tools like `pytest`.
        3. **Build and Deploy**: Once tests pass, the API is built and deployed to the server using Docker containers.
        4. **Monitoring**: Tools like Prometheus and Grafana are used to monitor API performance and model metrics.
    """)
    st.image('/Users/aleksandra.rancic/Desktop/Github.png')
    st.image('/Users/aleksandra.rancic/Desktop/Github2.png')
    st.image('/Users/aleksandra.rancic/Desktop/DockerHub.png')

elif page == "Further Improvements":
    st.title("Further Improvements to the Stock Prediction API")

    st.write("""
        There are several ways to further enhance the Stock Prediction API:
        
        1. **Feature Engineering**:
            - Creating new features: moving averages, exponential moving averages, volatility indices;
            - Involve technical indicators: Bollinger Bands, Moving average convergence divergence, Relative strength index - have predictive power in time series forecasting
        2. **Improving LSTM Model**: 
            - Enhancing the architecture of the model by introducing techniques like Bidirectional LSTM, increasing the depth of the LSTM network, improving Regularization techniques to avoid overfitting.
        3. **Ensemble Models**: Introduce ensemble models such as Random Forest or Gradient Boosting, which combine predictions from multiple models to improve accuracy.
        4. **Incorporate External Data**: Use external data sources like news sentiment analysis, economic indicators, and social media trends to enrich the dataset and improve model performance.
        5. **Improving Tests**: 
            - Enhance the test coverage for the API
        6. **User Authentication**: Implement user authentication and role-based access control (RBAC), allowing different levels of access (e.g., admin users can retrain the model).
        7. **Prometheus and Grafana Integration**: for real-time monitoring of the API's performance metrics and Grafana for data visualization. We can monitor the API response time, memory usage, and model prediction latency, and identify potential bottlenecks.
        
    """)

    st.subheader("Thank you for using Stock Prediction API")

