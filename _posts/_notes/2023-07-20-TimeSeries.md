---
title: Time Series
date: 2023-07-20 00:00:00 -0500
categories: [Core, Machine Learning]
tags: [forme, notes]
math: true
---

## Time Series and Forecasting

A sequence of measurements on the same variable over time made over regular intervals. Use of these sequences to create statistical model to predict future values. Forecasting is different from prediction because forecasting is dependent on sequential time related data.

Extrapolation is the process of inferring values outside the range of the existing data to make predictions. Extrapolation is one of the essential methods that data scientists use to predict future trends and outcomes.

### Components of a Time Series

- Trend
  - General direction of time series data
- Seasonality
  - Patterns of a time series where data experience regular and predictable changes every year.
- Cyclical Component
  - Patterns of a time series where data have a period of up and downs observed in business cycles but do not exhibit a seasonal variations
- Irregular/Residual Variation
  - Fluctuation in time series data that become evident when trend and seasonality are removed from the time series - unpredictable, erratic, may or may not be random

- Level
  - The baseline value for the series if it were a straight line
- Noise
  - The optional variability in the observation that cannot be explained by the mode

#### Concerns of Forecasting
- How much data do you have available and are you able to gather it all together? -> More data the better
- What is the time horizon of predictions that is required? Short, medium, or long term? -> Shorter times are easier to predict with higher confidence
- Can forecasts be updated frequently over time or must they be made once and remain static? -> Updating forecasts as new info often results in more accurate predictions
- At what temporal frequency are forecasts required? -> Able to forecasts at higher and lower frequencies, allows down-sampling and up-sampling

#### Data Preprocessing
Time Series often requires cleaning, scaling, and transformations
- Data with a frequency that is too high to model or is unevenly spaced through time requires resampling for some models
- Corrupt or extreme outlier values need to be identified and handled
- Gaps or missing data need to be interpolated or imputed

## Stationary and Non-Stationary Time Series

### Non-Stationary Time Series

- When time series have statistical properties of the time series change ove time. A time series containing a trend or seasonality is non-stationary

### Stationary Time Series

- When time series have a random process, where unconditional join probability distribution does not change with time
- A stationary time series has a constant variance that will return to the long term mean in n time stamps
- Has property of homoscedasticity - were the spread of data remains approximately the 
- Covariance is approximate constant

### Why do we convert a Non-Stationary Time Series to Stationary?

Forecasting a stationary time series is easier and more reliable than forecasting non-stationary. Some forecasting models are like linear regression models that uses lag of the series itself as predictor values. To solve the problem of correlation, converting to stationary series removes any persistent trend thereby making the predictors almost independent of each other.

Most models in TSA assume covariance-stationarity, and TS relies on stationarity or invalid otherwise

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

White Noise is not predictable, it is random. If you build a model and its residuals or post differencing looks like White noise, then you did as much as you can. If not there might be a better model. White noise is serially uncorrelated errors -> independent and identically distributed

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

### Linear Models
- Graphed as a straight line
- Assume serially uncorrelated errors
- Not often used in TS
$$ y_{t}=b_{0}+b_{1}t+\epsilon_{t} $$

```py
# simulate linear trend
# example Firm ABC sales are -$50 by default and +$25 at every time step

w = np.random.randn(100)
y = np.empty_like(w)

b0 = -50.
b1 = 25.
for t in range(len(w)):
    y[t] = b0 + b1*t + w[t]
    
_ = tsplot(y, lags=lags)  

```

### Log-Linear Models
- Graphed as exponential function
- Assume serially uncorrelated errors
- Not often used in TS
```py
# Simulate ABC exponential growth

# fake dates
idx = pd.date_range('2007-01-01', '2012-01-01', freq='M')

# fake sales increasing at exponential rate
sales = [np.exp( x/12 ) for x in range(1, len(idx)+1)]

# create dataframe and plot
df = pd.DataFrame(sales, columns=['Sales'], index=idx)

with plt.style.context('bmh'):
    df.plot()
    plt.title('ABC Sales')

# transform data by taking natural log of sales, now become linear regression

with plt.style.context('bmh'):
    pd.Series(np.log(sales), index=idx).plot()
    plt.title('ABC Log Sales')
```

### Simple Moving Average
- Naive approach to time series modeling
- Forecast next values in a time series based on average of a fixed finite number m of previous values
- States that the next observation is the mean of all past m observation
  $$ \hat{y}_{i}= \frac{1}{m}\sum_{j=i-m}^{i-1} y_{j}=\frac{y_{i-m}+...+y\_{i-1}}{m}$$

### Weighted Moving Average

- Weights of each previous m values can have different weights
- The sum of weights of all previous m values equal to 1
  $$ \hat{y}_{i}= w_{m}y_{i-m} + ... + w_{1}y\_{i-1} $$

### Autoregressive (AR) Model
- Tries to explain the momentum and mean reversion effects often observed in trading markets
- When dependent variables is regressed against one or more lagged values
- One of the simplest models for solving time series
- Value of $y$ at time $t$ depends on previous values
- Order of an autoregression is the number of previous values in the series used to predict
- P represents the number of lagged variables
- Coefficents cannot equal zero
  $$AR(p)\vdots \hat{y}_{i} = f(y_{t-1},y_{t-2},...y_{t-p})$$

```py
# Simulate an AR(1) process with alpha = 0.6

np.random.seed(1)
n_samples = int(1000)
a = 0.6
x = w = np.random.normal(size=n_samples)

for t in range(n_samples):
    x[t] = a*x[t-1] + w[t]
    
_ = tsplot(x, lags=lags)

# Fit an AR(p) model to simulated AR(1) model with alpha = 0.6

mdl = smt.AR(x).fit(maxlag=30, ic='aic', trend='nc')
%time est_order = smt.AR(x).select_order(
    maxlag=30, ic='aic', trend='nc')

true_order = 1
p('\nalpha estimate: {:3.5f} | best lag order = {}'
  .format(mdl.params[0], est_order))
p('\ntrue alpha = {} | true order = {}'
  .format(a, true_order))

# Simulate an AR(2) process

n = int(1000)
alphas = np.array([.666, -.333])
betas = np.array([0.])

# Python requires us to specify the zero-lag value which is 1
# Also note that the alphas for the AR model must be negated
# We also set the betas for the MA equal to 0 for an AR(p) model
# For more information see the examples at statsmodels.org
ar = np.r_[1, -alphas]
ma = np.r_[1, betas]

ar2 = smt.arma_generate_sample(ar=ar, ma=ma, nsample=n) 
_ = tsplot(ar2, lags=lags)

# Fit an AR(p) model to simulated AR(2) process

max_lag = 10
mdl = smt.AR(ar2).fit(maxlag=max_lag, ic='aic', trend='nc')
est_order = smt.AR(ar2).select_order(
    maxlag=max_lag, ic='aic', trend='nc')

true_order = 2
p('\ncoef estimate: {:3.4f} {:3.4f} | best lag order = {}'
  .format(mdl.params[0],mdl.params[1], est_order))
p('\ntrue coefs = {} | true order = {}'
  .format([.666,-.333], true_order))
  
# coef estimate: 0.6291 -0.3196 | best lag order = 2
# true coefs = [0.666, -0.333] | true order = 2

# Select best lag order for MSFT returns

max_lag = 30
mdl = smt.AR(lrets.MSFT).fit(maxlag=max_lag, ic='aic', trend='nc')
est_order = smt.AR(lrets.MSFT).select_order(
    maxlag=max_lag, ic='aic', trend='nc')

p('best estimated lag order = {}'.format(est_order))

# best estimated lag order = 23
```

### Moving Average (MA) Model
- Tries to explain the shock effect observed int he white noise terms. These shock effects could be thought of as unexpected events affecting the observation process
- Makes use of past error terms as opposed to the passed values themselves
- $\epsilon_{t}, \epsilon_{t-1}, ... \epsilon_{t-q}$ are past error terms
$$ Y_{t} = \beta_{0} + \epsilon_{t} +\phi_{1}\epsilon_{t-1}+\phi_{2}\epsilon_{t-2} + \phi_{3}\epsilon_{t-3}+ ... + \phi_{q}\epsilon\_{t-q}$$

```py
# Simulate an MA(1) process

n = int(1000)

# set the AR(p) alphas equal to 0
alphas = np.array([0.])
betas = np.array([0.6])

# add zero-lag and negate alphas
ar = np.r_[1, -alphas]
ma = np.r_[1, betas]

ma1 = smt.arma_generate_sample(ar=ar, ma=ma, nsample=n) 
_ = tsplot(ma1, lags=30)

# Fit the MA(1) model to our simulated time series
# Specify ARMA model with order (p, q)

max_lag = 30
mdl = smt.ARMA(ma1, order=(0, 1)).fit(
    maxlag=max_lag, method='mle', trend='nc')
p(mdl.summary())

# Simulate MA(3) process with betas 0.6, 0.4, 0.2

n = int(1000)
alphas = np.array([0.])
betas = np.array([0.6, 0.4, 0.2])
ar = np.r_[1, -alphas]
ma = np.r_[1, betas]

ma3 = smt.arma_generate_sample(ar=ar, ma=ma, nsample=n)
_ = tsplot(ma3, lags=30)

# Fit MA(3) model to simulated time series

max_lag = 30
mdl = smt.ARMA(ma3, order=(0, 3)).fit(
    maxlag=max_lag, method='mle', trend='nc')
p(mdl.summary())

# Fit MA(3) to SPY returns

max_lag = 30
Y = lrets.SPY
mdl = smt.ARMA(Y, order=(0, 3)).fit(
    maxlag=max_lag, method='mle', trend='nc')
p(mdl.summary())
_ = tsplot(mdl.resid, lags=max_lag)

```
### Auto-Regressive Moving Average (ARMA) Model

- Combo of AR and MA, where they are using bot past values and past error terms
- Ignores the volatility clustering effects found in most financial time series
$$Y_{t} = \beta_{0} + \beta_{1}Y_{t-1}+\beta_{2}Y_{t-2}+\beta_{3}Y_{t-3}+...+\beta_{p}Y_{t-p}+ \epsilon_{t}  +\phi_{1}\epsilon_{t-1} + \phi_{2}\epsilon_{t-2}+\phi_{3}\epsilon_{t-3}+ ...+ \phi_{q}\epsilon_{t-q}$$

```py
# Simulate an ARMA(2, 2) model with alphas=[0.5,-0.25] and betas=[0.5,-0.3]
max_lag = 30

n = int(5000) # lots of samples to help estimates
burn = int(n/10) # number of samples to discard before fit

alphas = np.array([0.5, -0.25])
betas = np.array([0.5, -0.3])
ar = np.r_[1, -alphas]
ma = np.r_[1, betas]

arma22 = smt.arma_generate_sample(ar=ar, ma=ma, nsample=n, burnin=burn)
_ = tsplot(arma22, lags=max_lag)

mdl = smt.ARMA(arma22, order=(2, 2)).fit(
    maxlag=max_lag, method='mle', trend='nc', burnin=burn)
p(mdl.summary())

# Simulate an ARMA(3, 2) model with alphas=[0.5,-0.25,0.4] and betas=[0.5,-0.3]

max_lag = 30

n = int(5000)
burn = 2000

alphas = np.array([0.5, -0.25, 0.4])
betas = np.array([0.5, -0.3])

ar = np.r_[1, -alphas]
ma = np.r_[1, betas]

arma32 = smt.arma_generate_sample(ar=ar, ma=ma, nsample=n, burnin=burn)
_ = tsplot(arma32, lags=max_lag)

# pick best order by aic 
# smallest aic value wins
best_aic = np.inf 
best_order = None
best_mdl = None

rng = range(5)
for i in rng:
    for j in rng:
        try:
            tmp_mdl = smt.ARMA(arma32, order=(i, j)).fit(method='mle', trend='nc')
            tmp_aic = tmp_mdl.aic
            if tmp_aic < best_aic:
                best_aic = tmp_aic
                best_order = (i, j)
                best_mdl = tmp_mdl
        except: continue


p('aic: {:6.5f} | order: {}'.format(best_aic, best_order))

# aic: 14108.27213 | order: (3, 2)

# Fit ARMA model to SPY returns

best_aic = np.inf 
best_order = None
best_mdl = None

rng = range(5) # [0,1,2,3,4,5]
for i in rng:
    for j in rng:
        try:
            tmp_mdl = smt.ARMA(lrets['SPY'], order=(i, j)).fit(
                method='mle', trend='nc'
            )
            tmp_aic = tmp_mdl.aic
            if tmp_aic < best_aic:
                best_aic = tmp_aic
                best_order = (i, j)
                best_mdl = tmp_mdl
        except: continue


p('aic: {:6.5f} | order: {}'.format(best_aic, best_order))

# aic: -11518.22902 | order: (4, 4)

```

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
```py
# Fit ARIMA(p, d, q) model to SPY Returns
# pick best order and final model based on aic

best_aic = np.inf 
best_order = None
best_mdl = None

pq_rng = range(5) # [0,1,2,3,4]
d_rng = range(2) # [0,1]
for i in pq_rng:
    for d in d_rng:
        for j in pq_rng:
            try:
                tmp_mdl = smt.ARIMA(lrets.SPY, order=(i,d,j)).fit(method='mle', trend='nc')
                tmp_aic = tmp_mdl.aic
                if tmp_aic < best_aic:
                    best_aic = tmp_aic
                    best_order = (i, d, j)
                    best_mdl = tmp_mdl
            except: continue


p('aic: {:6.5f} | order: {}'.format(best_aic, best_order))
# aic: -11518.22902 | order: (4, 0, 4)

# ARIMA model resid plot
_ = tsplot(best_mdl.resid, lags=30)

# Create a 21 day forecast of SPY returns with 95%, 99% CI
n_steps = 21

f, err95, ci95 = best_mdl.forecast(steps=n_steps) # 95% CI
_, err99, ci99 = best_mdl.forecast(steps=n_steps, alpha=0.01) # 99% CI

idx = pd.date_range(data.index[-1], periods=n_steps, freq='D')
fc_95 = pd.DataFrame(np.column_stack([f, ci95]), 
                     index=idx, columns=['forecast', 'lower_ci_95', 'upper_ci_95'])
fc_99 = pd.DataFrame(np.column_stack([ci99]), 
                     index=idx, columns=['lower_ci_99', 'upper_ci_99'])
fc_all = fc_95.combine_first(fc_99)
fc_all.head()

# Plot 21 day forecast for SPY returns

plt.style.use('bmh')
fig = plt.figure(figsize=(9,7))
ax = plt.gca()

ts = lrets.SPY.iloc[-500:].copy()
ts.plot(ax=ax, label='Spy Returns')
# in sample prediction
pred = best_mdl.predict(ts.index[0], ts.index[-1])
pred.plot(ax=ax, style='r-', label='In-sample prediction')

styles = ['b-', '0.2', '0.75', '0.2', '0.75']
fc_all.plot(ax=ax, style=styles)
plt.fill_between(fc_all.index, fc_all.lower_ci_95, fc_all.upper_ci_95, color='gray', alpha=0.7)
plt.fill_between(fc_all.index, fc_all.lower_ci_99, fc_all.upper_ci_99, color='gray', alpha=0.2)
plt.title('{} Day SPY Return Forecast\nARIMA{}'.format(n_steps, best_order))
plt.legend(loc='best', fontsize=10)
```

### Autoregressive Conditionally Heteroskedastic Models - ARCH(p)
- Like an AR model applied to the variance of a time series
- Variance of t of time is conditional on past observations

$$ Var(y_{t}|y_{t-1})= \sigma_{2}^{t} = \alpha_{0} + \alpha_{1}y_{t-1}^{2}$$
Assuming if series has zero mean
$$ y_{t} \sigma_{t}\epsilon_{t}, with \sigma_{t} =  \sqrt{\sigma_{0}+\sigma_{1}y_{t-1}^{2}}, and \epsilon_{t} ~ iid(0,1)$$

```py
# Simulate ARCH(1) series
# Var(yt) = a_0 + a_1*y{t-1}**2
# if a_1 is between 0 and 1 then yt is white noise

np.random.seed(13)

a0 = 2
a1 = .5

y = w = np.random.normal(size=1000)
Y = np.empty_like(y)

for t in range(len(y)):
    Y[t] = w[t] * np.sqrt((a0 + a1*y[t-1]**2))

# simulated ARCH(1) series, looks like white noise
tsplot(Y, lags=30)
```

### Generalized Autoregressive Conditionally Heteroskedastic Models - GARCH(p,q)
- An ARMA model applied to variance of time series - has both AR and MA term
$$\epsilon_{t} = \sigma_{t}w{t}$$
$$\sigma_{t}^{2} = \alpha_{0} + \alpha_{1}\epsilon_{t-1}^{2}+\beta_{1}\sigma_{t-1}^{2}$$
```py
# Simulating a GARCH(1, 1) process

np.random.seed(2)

a0 = 0.2
a1 = 0.5
b1 = 0.3

n = 10000
w = np.random.normal(size=n)
eps = np.zeros_like(w)
sigsq = np.zeros_like(w)

for i in range(1, n):
    sigsq[i] = a0 + a1*(eps[i-1]**2) + b1*sigsq[i-1]
    eps[i] = w[i] * np.sqrt(sigsq[i])

_ = tsplot(eps, lags=30)

# Fit a GARCH(1, 1) model to our simulated EPS series
# We use the arch_model function from the ARCH package

am = arch_model(eps)
res = am.fit(update_freq=5)
p(res.summary())
```

## Auto Correlation and Partial Auto Correlation Functions
### Serial Correlation
- When residuals errors of TS models are correlated to each other
- Stationarity TS are serially uncorrelated, if there is correlation there type 1 errors will occur
### Auto Correlation Function (ACF) and Serial Correlation
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

  
Resources:
[x] (What is Time Series Forecasting)[https://machinelearningmastery.com/time-series-forecasting/]  
[x] (A Gentle Introduction to Autocorrelation and Partial Autocorrelation)[https://machinelearningmastery.com/gentle-introduction-autocorrelation-partial-autocorrelation/]  
[x] (Time Series Analysis)[https://www.blackarbs.com/blog/time-series-analysis-in-python-linear-models-to-garch/11/1/2016] 
[x] (TSA)[https://www.quantstart.com/articles/#time-series-analysis]   
[x] (Comprehensive Guide To Time Series Forecast)[https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/]  
[] (ARIMA Model)[https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/]     

## Glossary Notation
$x$ = Random Variable  
$x_{t}$ = Random Variable at time t  
$h$ = Lag value  
$\forall$ = Indicates "for all"  
$E(X_{t})$ = Expected value of Stochastic process X  
$\mu$ = Mean of Stochastic process is constant value  
$t_{1}, t_{2}= t_{1} and t_{2}$ = are the two different timestamps  
$\sum$ = Summation  
$R_{x}(t_{1},t_{2})$ = AutoCovariance  
$\lambda$ = Window size of the time series, the number of data points which we choose  
$N$ = The total number of samples  
$\hat{\mu}$ = The summation of X values ranging from lambda to N - 1 divided by the subtraction of total number of samples and lambda  
$\tau = t_{1}-t_{2}$ = difference between the past value and the present value  
$\hat{R}_{x}(\tau)$ = Sample autocovariance for each lambda  
$S_{t}$ = Seasonal Component of the time series  
$k$ = Seasonality period  
$y_{t}$ = Sequence of random variables  
$\hat{Y_{t}}$ = Time series after applying smoothing (removing the fine-grained variation between time stamps)  
$y_{h}$ = The periodic regression coefficient of $S_{t+h}$ in order to remove seasonality from the data  
$w_{t}$ = White Noise  
$\sigma^{2}$ = Variance  
$\delta_{t1-t2}$ = Delta change in time  
$p$ = The number of past orders to be included in the Auto Regressive AR model
$a_{i}$ = Coefficients of hte Auto Regressive model  
$z$ = Variable of the polynomial  
$b_{i}$ = Coefficients of the Moving Average model  
$q$ = The order of the Moving Average model  
$x_{t}-X_{t-1}$ = First Order differencing  
$A(z)=A(z)$ = is a matrix where each row acts as a regressor   
$|| ||$ = Denotes the norm of a vector    
