# Pair Trading Strategy

It is a pair trading strategy on a bucket of US stocks.

It is written in the [Quantopian](https://www.quantopian.com/home) platform. It is written in python and if you want to run it without using Quantopian, you can install zipline package and the program will work fine. (Although few more actions need to be made for downloading the stocks data).

## Stock selection
Create a bucket of stocks that are fundamentally correlated to each other (eg GOOG and GOOGL, V and MA). Also add stocks where their prices are statistically correlated. 

## Strategy
Pair trading a pair of highly correlated/cointegrated stocks. If the price of the two stocks abnormally deviate from each other, we  believe the pair would convert to each others in terms of price in the near future. (It is called mean-reversion process). We short sell the increasing one and long the decreasing one when they deviate from each other. If we enter position of the pair, we use market neutral approach instead of money neutral to avoid any extra exposure of the position.

The focus on this strategy is to ensure the risk is minimal, and we take profit while we can. So the expected return is not high, and the expected volatility is low.

## Work Flow
First use OLS regression to calculate the beta(hedge ratio) of the pair.

Using the hedge ratio, we can calculate the spread everyday and create a series of spread using previous spread calulcated. 

From the spread series, we perform 3 tests on the series sequentially, if any two of the tests fail, we would not trade the pair or exit the position of the pair.

If the pair pass through the test stage, we would calculate the standard score of the spread of the day to decide the direction of position of the pair. Using the hedge ratio, we decide the ratio of the position


## The 3 Tests
Perform 3 tests to ensure the existence of cointegration of the price of the pair.

### 1. ADF test
Perform ADF test on the spread of the pair. If the p value is acceptable, we believe the spread is stationary. Thus the price of the pair of the stocks are cointegrated and the ADF test will be passed.


### 2. Half Life Test
Half life is to calculate the duration for a value to decrease to half of its value. Using the half life algorithm on the spread, we can calculate the period of the mean reversion cycle. If the mean revision process takes too long to be complete, the half life test would be failed.

### 3. Hurst Exponent
Hurst exponent is a measure of the long term memory of a time series. It is similar to the autocorrelation but it emphasizes on the long term autocorrelation. Denote Hurst Exponent as H

if 0 < H < 0.5, we believe the time series has long term dependence and high and low values would be switching in adjacent pairs.

if 0.5 < H < 1, we believe the time series has no long term dependence and the value of the time series would be a trend.

## Performance
2015-2019

Total Return = 6%

Beta = 0.04
