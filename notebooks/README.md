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
  - ![BND Closing Price](../figures/BND%20Closing%20Price%20Over%20Time.png)

- **SPY Closing Price**: 
  - The price showed a consistent upward trend, indicating strong performance over the analyzed period.
  - ![SPY Closing Price](../figures/SPY%20Closing%20Price%20Over%20Time.png)

- **TSLA Closing Price**: 
  - TSLA displayed significant volatility, with a general upward trend followed by fluctuations.
  - ![TSLA Closing Price](../figures/TSLA%20Closing%20Price%20Over%20Time.png)

### Fluctuations in Daily Returns
- **BND Daily Returns**: 
  - Fluctuations were observed, particularly around 2020.
  - ![BND Daily Returns](../figures/BND%20Daily%20Percentage%20Change%20Over%20Time.png)

- **SPY Daily Returns**: 
  - Similar fluctuations were noted, with some extreme values during market events.
  - ![SPY Daily Returns](../figures/SPY%20Daily%20Percentage%20Change%20Over%20Time.png)

- **TSLA Daily Returns**: 
  - TSLA showed considerable volatility, with several unusual return days highlighted.
  - ![TSLA Daily Returns](../figures/TSLA%20Daily%20Percentage%20Change%20Over%20Time.png)

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
  - ![BND Trend](../figures/BND%20Trend.png)

- **SPY**:
  - ![SPY Trend](../figures/SPY%20Trend.png)

- **TSLA**:
  - ![TSLA Trend](../figures/TSLA%20Trend.png)

### Conclusion
The analysis provides a comprehensive view of the stock performance of Tesla, BND, and SPY. It highlights key trends, fluctuations in returns, and evaluates risk through VaR and Sharpe Ratio metrics.
