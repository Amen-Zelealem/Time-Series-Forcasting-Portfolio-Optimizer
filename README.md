# 📈 Time-Series Forecasting Portfolio Optimizer

This repository contains a comprehensive framework for applying time series forecasting techniques to optimize portfolio management strategies. The project is designed for financial analysts at Guide Me in Finance (GMF) Investments, focusing on leveraging historical financial data to enhance investment decision-making.

## ✨ Key Features Include:

- **📊 Data Acquisition**: Utilizes the YFinance Python library to fetch historical stock prices and financial metrics for key assets such as Tesla (TSLA), Vanguard Total Bond Market ETF (BND), and S&P 500 ETF (SPY) from January 1, 2015, to January 31, 2025.
  
- **🔧 Data Preprocessing**: Implements data cleaning, normalization, and exploratory data analysis (EDA) to prepare datasets for modeling.
  
- **🔮 Forecasting Models**: Develops and evaluates various time series forecasting models, including ARIMA, SARIMA, and LSTM, to predict future market trends.
  
- **📈 Portfolio Optimization**: Analyzes forecasted data to recommend portfolio adjustments aimed at maximizing returns while managing risks, using metrics such as Sharpe Ratio and Value at Risk (VaR).
  
- **📉 Visualization Tools**: Provides visualizations for historical data, forecast results, and portfolio performance to facilitate data-driven decision-making.


# Completed Task


# Data Preprocessing for Stock Analysis

## Overview
This document outlines the data preprocessing steps taken for analyzing the stock prices of Tesla (TSLA), BND, and SPY. It includes data inspection, basic statistics, and checks for missing values and duplicates.

## Data Inspection

### BND Data
🔍 **First 10 Rows of BND Data:**
| Date       | Open  | High    | Low     | Close  | Volume    |
|------------|-------|---------|---------|--------|-----------|
| 2025-01-31 | 72.48 | 72.5400 | 72.2000 | 72.34  | 6738335.0 |
| 2025-01-30 | 72.46 | 72.5300 | 72.3800 | 72.44  | 5622434.0 |
| 2025-01-29 | 72.43 | 72.4900 | 72.1700 | 72.34  | 5780349.0 |
| 2025-01-28 | 72.39 | 72.3900 | 72.2100 | 72.38  | 4424518.0 |
| 2025-01-27 | 72.34 | 72.4400 | 72.2600 | 72.42  | 8621175.0 |
| 2025-01-24 | 72.00 | 72.0943 | 71.8696 | 72.04  | 5555722.0 |
| 2025-01-23 | 72.00 | 72.0000 | 71.8000 | 71.90  | 7529793.0 |
| 2025-01-22 | 72.15 | 72.2000 | 71.9700 | 72.01  | 6616809.0 |
| 2025-01-21 | 72.21 | 72.2100 | 72.0725 | 72.16  | 8491622.0 |
| 2025-01-17 | 72.10 | 72.1000 | 71.9100 | 71.95  | 5600397.0 |

📋 **Data Inspection Results:**
- Data Types:
Open float64
High float64
Low float64
Close float64
Volume float64
dtype: object

- Missing Values: There are no Missing Values 
- Duplicate Rows: There is also No Duplicates

The data preprocessing steps included thorough inspection for data types, missing values, and duplicate rows. Basic statistics were also computed to summarize the dataset effectively, preparing it for further analysis.

# Stock Analysis of Tesla (TSLA), BND, and SPY

## Overview
This document summarizes the analysis of Tesla's stock price (TSLA), along with BND and SPY. It covers key insights including overall trends, fluctuations in daily returns, Value at Risk (VaR), and Sharpe Ratio to assess potential losses and risk-adjusted returns.

## Key Insights

### Overall Direction of Stock Prices
- **BND Closing Price**: 
  - The price exhibited fluctuations, with notable trends from 2016 to 2024.
  - ![BND Closing Price](/figures/BND%20Closing%20Price%20Over%20Time.png)

- **SPY Closing Price**: 
  - The price showed a consistent upward trend, indicating strong performance over the analyzed period.
  - ![SPY Closing Price](/figures/SPY%20Closing%20Price%20Over%20Time.png)

- **TSLA Closing Price**: 
  - TSLA displayed significant volatility, with a general upward trend followed by fluctuations.
  - ![TSLA Closing Price](/figures/TSLA%20Closing%20Price%20Over%20Time.png)

### Fluctuations in Daily Returns
- **BND Daily Returns**: 
  - Fluctuations were observed, particularly around 2020.
  - ![BND Daily Returns](/figures/BND%20Daily%20Percentage%20Change%20Over%20Time.png)

- **SPY Daily Returns**: 
  - Similar fluctuations were noted, with some extreme values during market events.
  - ![SPY Daily Returns](/figures/SPY%20Daily%20Percentage%20Change%20Over%20Time.png)

- **TSLA Daily Returns**: 
  - TSLA showed considerable volatility, with several unusual return days highlighted.
  - ![TSLA Daily Returns](/figures/TSLA%20Daily%20Percentage%20Change%20Over%20Time.png)

### Value at Risk (VaR) and Sharpe Ratio
- **VaR Analysis**: 
  - VaR was calculated to assess potential losses over a specified time frame for each stock.

- **Sharpe Ratio**: 
  - The Sharpe Ratio was calculated for each stock to evaluate risk-adjusted returns, indicating the effectiveness of returns relative to risk.

### ADF Test Results
- **BND**: 
  - ADF test p-value: 0.7333 (non-stationary)
  - After differencing: p-value: 1.5468e-26

- **SPY**: 
  - ADF test p-value: 0.2809 (non-stationary)
  - After differencing: p-value: 4.6835e-29

- **TSLA**: 
  - ADF test p-value: 0.0739 (non-stationary)
  - After differencing: p-value: 4.6503e-17

### Trend and Seasonality Analysis
- **BND**:
  - ![BND Trend](/figures/BND%20Trend.png)

- **SPY**:
  - ![SPY Trend](/figures/SPY%20Trend.png)

- **TSLA**:
  - ![TSLA Trend](/figures/TSLA%20Trend.png)

### Conclusion
The analysis provides a comprehensive view of the stock performance of Tesla, BND, and SPY. It highlights key trends, fluctuations in returns, and evaluates risk through VaR and Sharpe Ratio metrics.


# 📁 **Project Structure**

```
+---.github
| └── workflows
|   ├──  blank.yml
+---.vscode
| └── settings.json
+---figures
| ├── BND Closing Price Over Time.png
| ├── BND Daily Percentage Change Over Time.png
| ├── BND Trend.png
| ├── Closing Price Trend, Rolling Mean and Volatility of BND.png
| ├── Closing Price Trend, Rolling Mean and Volatility of SPY.png
| ├── Closing Price Trend, Rolling Mean and Volatility of TSLA.png
| ├── Daily Returns With Unusual Days Highlighted - BND.png
| ├── Daily Returns With Unusual Days Highlighted - SPY.png
| ├── Daily Returns With Unusual Days Highlighted - TSLA.png
| ├── SPY Closing Price Over Time.png
| ├── SPY Daily Percentage Change Over Time.png
| ├── SPY Trend.png
| ├── TSLA Closing Price Over Time.png
| ├── TSLA Daily Percentage Change Over Time.png
| └── TSLA Trend.png

+---notebooks
| ├── __init__.ipynb
| ├──data_preprocessing.ipynb
| ├──exploratory_data_analysis.ipynb
| └── README.md
+---scripts
| ├── init.py
| ├── data_preprocessing.py
| ├── exploratory_data_analysis.py
| └── README.md
+---src
| ├── init.py
| └── README.md
+---tests
| ├── init.py
| ├── README.md
| ├── .gitignore
| ├── LICENSE
| ├── README.md
| └── requirements.txt
```
