# Ex.No: 07                                       AUTO REGRESSIVE MODEL
### Date: 25-05-26



### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM : 
```
"""
AIM:
To implement an Auto Regressive (AR) Model using Python on the Nifty 50
stock dataset (HDFCBANK.NS) to forecast monthly average close prices,
evaluate stationarity using the Augmented Dickey-Fuller test, and assess
model performance using Mean Squared Error (MSE).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error

# ── 1. Read CSV into DataFrame ────────────────────────────────────────────────
STOCK    = 'HDFCBANK.NS'
CSV_PATH = 'nifty50_2000_2025.csv'   # keep CSV in same folder as notebook

df = pd.read_csv(CSV_PATH)
df['Date'] = pd.to_datetime(df['Date'])

stock_df = df[df['Stock'] == STOCK].sort_values('Date').copy()
stock_df['YearMonth'] = stock_df['Date'].dt.to_period('M')

monthly = stock_df.groupby('YearMonth')['Close'].mean()
monthly.index = monthly.index.to_timestamp()
monthly.index.freq = 'MS'

# ── GIVEN DATA ────────────────────────────────────────────────────────────────
print("=" * 55)
print("GIVEN DATA")
print("=" * 55)
print(f"\nStock       : {STOCK}")
print(f"Data points : {len(monthly)}  (monthly avg close, 2000–2024)")
print(f"Date range  : {monthly.index[0].date()} → {monthly.index[-1].date()}")
print(f"\nFirst 5 rows:\n{monthly.head().to_frame()}")
print(f"\nBasic Stats:\n{monthly.describe().round(2)}")

plt.figure(figsize=(14, 4))
plt.plot(monthly, color='steelblue', linewidth=1.5, label='Monthly Avg Close')
plt.title(f'{STOCK} — Monthly Avg Close Price (2000–2024)')
plt.xlabel('Date')
plt.ylabel('Close Price (₹)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ── 2. Augmented Dickey-Fuller Test ───────────────────────────────────────────
print("\n" + "=" * 55)
print("AUGMENTED DICKEY-FULLER TEST")
print("=" * 55)
adf_result = adfuller(monthly)
print(f"ADF Statistic : {adf_result[0]:.4f}")
print(f"p-value       : {adf_result[1]:.4f}")
print(f"Critical Values:")
for key, val in adf_result[4].items():
    print(f"   {key} : {val:.4f}")
if adf_result[1] > 0.05:
    print("\nConclusion: Series is NON-STATIONARY (p > 0.05) → differencing needed")
else:
    print("\nConclusion: Series is STATIONARY (p < 0.05)")

# ── 3. Train / Test Split ─────────────────────────────────────────────────────
train = monthly[:-12]
test  = monthly[-12:]

print(f"\nTrain : {len(train)} months  ({train.index[0].date()} → {train.index[-1].date()})")
print(f"Test  : {len(test)} months   ({test.index[0].date()} → {test.index[-1].date()})")

# ── 4. Fit AR Model with 13 lags ──────────────────────────────────────────────
LAGS = 13
ar_model = AutoReg(train, lags=LAGS).fit()

print(f"\nAR({LAGS}) Model fitted successfully.")
print(ar_model.summary())

# ── 5. PACF and ACF Plots ─────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("PACF - ACF")
print("=" * 55)

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

plot_pacf(monthly, lags=40, ax=axes[0])
axes[0].set_title(f'{STOCK} — Partial Autocorrelation Function (PACF)')
axes[0].set_xlabel('Lag (months)')
axes[0].grid(alpha=0.3)

plot_acf(monthly, lags=40, ax=axes[1])
axes[1].set_title(f'{STOCK} — Autocorrelation Function (ACF)')
axes[1].set_xlabel('Lag (months)')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ── 6. Make Predictions on Test Set ──────────────────────────────────────────
start = len(train)
end   = len(train) + len(test) - 1
test_pred = ar_model.predict(start=start, end=end)
test_pred.index = test.index

# ── 7. Calculate MSE ──────────────────────────────────────────────────────────
mse  = mean_squared_error(test, test_pred)
rmse = np.sqrt(mse)
print(f"\nMSE  (Test) : {mse:.2f}")
print(f"RMSE (Test) : {rmse:.2f}")

# ── PREDICTION PLOT ───────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("PREDICTION")
print("=" * 55)

plt.figure(figsize=(14, 5))
plt.plot(train,     color='steelblue', linewidth=1.5,               label='Train Data')
plt.plot(test,      color='green',     linewidth=1.5,               label='Actual Test')
plt.plot(test_pred, color='tomato',    linewidth=1.5, linestyle='--', label=f'AR({LAGS}) Predicted (RMSE={rmse:.2f})')
plt.title(f'{STOCK} — AR({LAGS}) Test Prediction vs Actual')
plt.xlabel('Date')
plt.ylabel('Close Price (₹)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ── 8. Fit on Full Data → Future Prediction (12 months) ──────────────────────
final_ar = AutoReg(monthly, lags=LAGS).fit()
future_pred = final_ar.predict(start=len(monthly), end=len(monthly) + 11)
future_index = pd.date_range(start=monthly.index[-1], periods=13, freq='MS')[1:]
future_series = pd.Series(future_pred.values, index=future_index)

print("\n" + "=" * 55)
print("FINAL PREDICTION")
print("=" * 55)
print("\nFuture 12-Month Forecast (Jan 2025 – Dec 2025):")
print(future_series.round(2).to_frame(name='Predicted Close (₹)'))

plt.figure(figsize=(14, 5))
plt.plot(monthly,       color='steelblue', linewidth=1.5,               label='Historical Data')
plt.plot(future_series, color='tomato',    linewidth=2,   linestyle='--', label='Future Forecast (12 months)')
plt.axvline(x=monthly.index[-1], color='gray', linestyle=':', linewidth=1.2, label='Forecast Start')
plt.title(f'{STOCK} — AR({LAGS}) Final Prediction (Next 12 Months)')
plt.xlabel('Date')
plt.ylabel('Close Price (₹)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("\nRESULT:")
print("Thus the program ran successfully based on the Auto Regressive (AR) Model.")

```
### OUTPUT:

GIVEN DATA
```
Stock       : HDFCBANK.NS
Data points : 299  (monthly avg close, 2000–2024)
Date range  : 2000-02-01 → 2024-12-01

First 5 rows:
                Close
YearMonth            
2000-02-01  11.551923
2000-03-01  13.150000
2000-04-01  11.367125
2000-05-01  12.426304
2000-06-01  12.466023

Basic Stats:
count    299.00
mean     270.91
std      279.45
min        9.72
25%       37.88
50%      145.32
75%      500.53
max      913.28
Name: Close, dtype: float64
```
<img width="1123" height="319" alt="image" src="https://github.com/user-attachments/assets/52a86665-f757-474c-bb08-dab49d921037" />


PACF - ACF
```
=======================================================
AUGMENTED DICKEY-FULLER TEST
=======================================================
ADF Statistic : 2.4411
p-value       : 0.9990
Critical Values:
   1% : -3.4537
   5% : -2.8718
   10% : -2.5722

Conclusion: Series is NON-STATIONARY (p > 0.05) → differencing needed

Train : 287 months  (2000-02-01 → 2023-12-01)
Test  : 12 months   (2024-01-01 → 2024-12-01)

AR(13) Model fitted successfully.
                            AutoReg Model Results                             
==============================================================================
Dep. Variable:                  Close   No. Observations:                  287
Model:                    AutoReg(13)   Log Likelihood               -1137.626
Method:               Conditional MLE   S.D. of innovations             15.379
Date:                Mon, 25 May 2026   AIC                           2305.253
Time:                        09:16:34   BIC                           2359.450
Sample:                    03-01-2001   HQIC                          2327.006
                         - 12-01-2023                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
const          1.8304      1.310      1.398      0.162      -0.736       4.397
Close.L1       1.3425      0.061     21.923      0.000       1.222       1.463
Close.L2      -0.5844      0.102     -5.734      0.000      -0.784      -0.385
Close.L3       0.2943      0.106      2.766      0.006       0.086       0.503
Close.L4      -0.1510      0.105     -1.431      0.152      -0.358       0.056
Close.L5       0.1747      0.105      1.661      0.097      -0.031       0.381
Close.L6      -0.3487      0.106     -3.284      0.001      -0.557      -0.141
Close.L7       0.3969      0.107      3.711      0.000       0.187       0.607
Close.L8      -0.1266      0.109     -1.159      0.246      -0.341       0.087
Close.L9      -0.2988      0.111     -2.693      0.007      -0.516      -0.081
Close.L10      0.4542      0.114      3.977      0.000       0.230       0.678
Close.L11     -0.3484      0.116     -2.994      0.003      -0.576      -0.120
Close.L12      0.2321      0.110      2.102      0.036       0.016       0.449
Close.L13     -0.0276      0.066     -0.421      0.674      -0.156       0.101
                                    Roots                                     
==============================================================================
                   Real          Imaginary           Modulus         Frequency
------------------------------------------------------------------------------
AR.1            -1.1683           -0.0000j            1.1683           -0.5000
AR.2            -0.8323           -0.6329j            1.0456           -0.3965
AR.3            -0.8323           +0.6329j            1.0456            0.3965
AR.4             0.9934           -0.0000j            0.9934           -0.0000
AR.5             1.0321           -0.4008j            1.1072           -0.0590
AR.6             1.0321           +0.4008j            1.1072            0.0590
AR.7             0.7168           -0.9331j            1.1766           -0.1457
AR.8             0.7168           +0.9331j            1.1766            0.1457
AR.9            -0.2170           -1.1514j            1.1716           -0.2797
AR.10           -0.2170           +1.1514j            1.1716            0.2797
AR.11            0.1475           -1.3255j            1.3337           -0.2324
AR.12            0.1475           +1.3255j            1.3337            0.2324
AR.13            6.8784           -0.0000j            6.8784           -0.0000
------------------------------------------------------------------------------

```
<img width="1121" height="714" alt="image" src="https://github.com/user-attachments/assets/b3e98f9e-9239-4376-b78c-2d805daa231f" />




PREDICTION 
<img width="1119" height="410" alt="image" src="https://github.com/user-attachments/assets/63ad9b2e-92a5-49ed-9837-482097bd77d4" />


FINIAL PREDICTION
```
Future 12-Month Forecast (Jan 2025 – Dec 2025):
            Predicted Close (₹)
2025-01-01               917.77
2025-02-01               920.26
2025-03-01               911.16
2025-04-01               910.36
2025-05-01               914.66
2025-06-01               912.85
2025-07-01               920.17
2025-08-01               922.33
2025-09-01               921.85
2025-10-01               927.02
2025-11-01               932.71
2025-12-01               944.74
```
<img width="1120" height="408" alt="image" src="https://github.com/user-attachments/assets/d77cfa86-6368-4f51-a0f2-43e2bb35c2f4" />


### RESULT:
Thus we have successfully implemented the auto regression function using python.
