---
title: Time Series
date: 2023-07-20 00:00:00 -0500
categories: [Core, Machine Learning]
tags: [forme, notes]
math: true
---

## Time Series and Forecasting

A sequence of measurements on the same variable over time made over regular intervals. Use of these sequences to create statistical model to predict future values. Forecasting is different from prediction because forecasting is dependent on sequential time related data.

### Components of a Time Series

- Trend
  - General direction of time series data
- Seasonality
  - Patterns of a time series where data experience regular and predictable changes every year.
- Cyclical Component
  - Patterns of a time series where data have a period of up and downs observed in business cycles but do not exhibit a seasonal variations
- Irregular/Residual Variation
  - Fluctuation in time series data that become evident when trend and seasonality are removed from the time series - unpredictable, erratic, may or may not be random

## Stationary and Non-Stationary Time Series

### Non-Stationary Time Series

- When time series have statistical properties of the time series change ove time. A time series containing a trend or seasonality is non-stationary

### Stationary Time Series

- When time series have a random process, where unconditional join probability distribution does not change with time
- A stationary time series has a constant variance that will return to the long term mean in n time stamps

### Why do we convert a Non-Stationary Time Series to Stationary?

Forecasting a stationary time series is easier and more reliable than forecasting non-stationary. Some forecasting models are like linear regression models that uses lag of the series itself as predictor values. To solve the problem of correlation, converting to stationary series removes any persistent trend thereby making the predictors almost independent of each other.

### Detecting the Stationarity of a Time Series

1. Plot data
2. Data points will always return towards long run mean with a constant variance
   - We can also use Augmented Dickey-Fuller ADF test to check

### How to Make a Time Series Stationary

Most common method is by differencing the series, until it becomes approximately stationary.  
$$ Y*{t}-Y*{t-1} $$  
Subtract previous value $Y_{t-1}$ with current value $ Y\_{t}$  
Keep performing differencing until it's stationary,the number of time is called difference order

The first value would be removed because it will be NAN

## White Noise and Random Walk

ML doesn't perform well in time series forecasting because of White Noise and Random Walks.

### White Noise

White Noise is not predictable, it is random. If you build a model and its residuals or post differencing looks like White noise, then you did as much as you can. If not there might be a better model.

For a time series to be categorized as White Noise:

1. The mean value should be zero
2. The standard deviation should be constant, it shouldn't change over time
3. There must be zero auto correlation at all lags

Methods for determining if time series resembles White Noise:

1. Compare mean and standard deviation over time
2. Plot the time series
3. Examine autocorrelation plots

### Random Walk

In Random Walk each value is dependent on the previous value with some noise. It difficult to forecast random walks because using previous value does not ensure results, and the addition of white noise makes it more difficult.

To make a dummy Random Walk series:

1. Begin with an arbitrary value, such as zero
2. The next value is the previous value plus some added random fluctuation. You can go through the procedure of adding more values as many times you like.
   Math Random Noise:  
   $$X_{t}= X_{t-1} + W_{t} $$

## Forecasting Methods

### Simple Moving Average

- Naive approach to time series modeling
- Forecast next values in a time series based on average of a fixed finite number m of previous values
- States that the next observation is the mean of all past m observation
  $$ \hat{y}_{i}= \frac{1}{m}\sum_{j=i-m}^{i-1} y*{j}=\frac{y*{i-m}+...+y\_{i-1}}{m}$$

### Weighted Moving Average

- Weights of each previous m values can have different weights
- The sum of weights of all previous m values equal to 1
  $$ \hat{y}_{i}= w_{m}y*{i-m} + ... + w*{1}y\_{i-1} $$

### Autoregressive (AR) Model

- One of the simplest models for solving time series
- Value of $y$ at time $t$ depends on previous values
- Order of an autoregression is the number of previous values in the series used to predict
  $$AR(p)\vdots \hat{y}_{i} = f(y_{t-1},y_{t-2},...y_{t-p})$$

### Moving Average (MA) Model

- Makes use of past error terms as opposed to the passed values themselves
- $\epsilon_{t}, \epsilon_{t-1}, ... \epsilon_{t-q}$ are past error terms
$$ Y_{t} = \beta_{0} + \epsilon_{t} +\phi_{1}\epsilon_{t-1}+\phi_{2}\epsilon_{t-2} + \phi_{3}\epsilon_{t-3}+ ... + \phi_{q}\epsilon\_{t-q}$$

### Auto-Regressive Moving Average (ARMA) Model

- Combo of AR and MA, where they are using bot past values and past error terms
$$Y_{t} = \beta_{0} + \beta_{1}Y_{t-1}+\beta_{2}Y_{t-2}+\beta_{3}Y_{t-3}+...+\beta_{p}Y_{t-p}+ \epsilon_{t}  +\phi_{1}\epsilon_{t-1} + \phi_{2}\epsilon_{t-2}+\phi_{3}\epsilon_{t-3}+ ...+ \phi_{q}\epsilon_{t-q}$$

### Auto-Regressive Integrated Moving Average (ARIMA) Model
- Very popular model
- Generalized ARMA model that uses integration for attaining stationarity
- Made of 3 parameters
    - p: Lag order or the number of past orders to be included in the model, i.e., the order of AutoRegression
        - Determined from the number of time the data has had past values subtracted
        - ARIMA (1,0,0): $y_{t} = a_{1}y_{t-1}+\epsilon_{t}$
    - d: The degree of differencing to be applied (the number of times the data has had past values subtracted)
        - Use ACF plots to deteremine value
        - ARIMA (2,0,0): $y_{t} =  a_{1}y_{t-1}+ a_{2}y_{t-2}+\epsilon_{t}$
    - q: The moving average
        - Use PACF plots to deteremine value
        - ARIMA (2,1,1): $\Delta y_{t} = a_{1}\Delta y_{t-1}+ a_{2}\Delta y_{t-2}+b_{1}\epsilon_{t-1}$ 
        - $ where \Delta y_{t} = \Delta y_{t}-\Delta y_{t-1}$

## Auto Correlation and Partial Auto Correlation Functions
### Auto Correlation Function (ACF)
- ACF plot is a bar chart of coefficients of correlation between the time series and lagged values
- Given $y$ is value and $t$ is time, then the correlation between $y^(t)$ and $y^(t-1)$ is lag
- From this plot we can graph the confidence interval where there is a significance of correlation 
- Estimation of q in MA model
    - We use ACF plot to estimate MA(q)
    - Observe how many lags are above or below confidence interval before next lag enters the blue area
### Partial Auto Correlation Function (PACF)
- Describes partial correlation between the series and its own lag
- For each lag there is a unique correlation between those two observation after taking out the intervening correlation
- Where there is a lag not explained by other lags
- Estimation of p in AR model
    - First ignore value at lag 0
        - Will show perfect correlation since we are estimating correlation between present value and itself
    - Calculate lags are above or below the confidence interval before the next lag enters the confidence interval