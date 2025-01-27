**Inventory Demand Forecasting**

**Overview**

This project implements an inventory demand forecasting system to predict future product demand based on historical sales data. It uses time series analysis and machine learning models to ensure optimal inventory management, reduce costs, and improve supply chain efficiency.

**Features**

1. Time Series Forecasting: Implements methods like ARIMA, Prophet, and LSTM for predicting future demand.
  
2. Exploratory Data Analysis (EDA): Visualizes patterns, trends, and seasonality in sales data.
   
3. Data Preprocessing: Handles missing values, outliers, and ensures data quality.
   
4. Model Evaluation: Uses metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).
   
5. Scalability: Supports forecasting for multiple products or categories.

**Dataset**

**Dataset Details**

1. Name: Perrin Freres Monthly Champagne Sales
   
2. Description: Historical monthly sales data of champagne products.
   
3. File: perrin-freres-monthly-champagne-.csv

4. Columns:
    Month: The time period (YYYY-MM format).
    Sales: Monthly sales figures.

**Installation**

**Prerequisites**

Python 3.8 or higher

Libraries:

  pandas
  
  numpy
  
  matplotlib
  
  seaborn
  
  scikit-learn
  
  statsmodels
  
  tensorflow (if using LSTMs)
  
  fbprophet (optional for Prophet model)

**Steps**
1. Clone the repository:
   git clone https://github.com/your-username/inventory-demand-forecasting.git
   cd inventory-demand-forecasting

2. Install the required dependencies:
  pip install -r requirements.txt

3. Run the Jupyter Notebook or script:
  jupyter notebook InventoryDemandForecasting.ipynb

**Usage**
1. Exploratory Data Analysis (EDA):
    Visualize historical sales trends and seasonal patterns.
   
2. Data Preprocessing:
    Clean and prepare data for modeling.
   
3. Forecasting Models:
    Implement and evaluate forecasting models such as ARIMA, LSTM, or Prophet.
   
4. Evaluation Metrics:
    Assess model performance using MAE, MSE, and RMSE.

**Examples**
**Example: Visualizing Sales Trends**

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('perrin-freres-monthly-champagne-.csv')
data['Month'] = pd.to_datetime(data['Month'])
data.set_index('Month', inplace=True)

plt.plot(data['Sales'])
plt.title('Monthly Champagne Sales')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.show()

**Example: Running an ARIMA Model**

from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(data['Sales'], order=(5, 1, 0))
model_fit = model.fit()
forecast = model_fit.forecast(steps=12)
print(forecast)

**Results**

1. Visualized trends, seasonality, and patterns in historical data.
2. Generated accurate forecasts using ARIMA, LSTM, and Prophet.
3. Improved inventory management by reducing stock-outs and overstocking.

**Challenges and Solutions**

1. Handling Missing Data:
    Solution: Used linear interpolation and forward-fill methods to address missing values.

2. Model Selection:
    Solution: Performed hyperparameter tuning and compared models using cross-validation.

3. Scalability:
    Solution: Designed the system to forecast multiple SKUs or categories efficiently.

**Future Work**
1. Extend the system to include external features such as promotions, holidays, and weather data.
2. Implement advanced neural network models like Transformers for improved accuracy.
3. Develop a web interface for interactive forecasting and visualization.

References
  ARIMA Model Documentation
  Prophet Documentation
  TensorFlow LSTM Guide
