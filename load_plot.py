# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 16:39:37 2024

@author: fahad
"""

import pandas as pd
import numpy as np
import datetime
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from typing import Optional
from bayes_linear_regression import BayesianLinearRegression, expand

def load_temperature_data(filename: str, year: Optional[int] = None) -> pd.DataFrame:
    names = ['station', 'date', 'type', 'measurement', 'e1', 'e2', 'E', 'e3']
    data = pd.read_csv(filename, names=names)
    data['date'] = pd.to_datetime(data['date'], format="%Y%m%d")
    tmax = data[data['type'] == 'TMAX'][['date', 'measurement']]
    tmin = data[data['type'] == 'TMIN'][['date', 'measurement']]
    prcp = data[data['type'] == 'PRCP'][['date', 'measurement']]
    arr = np.array([tmax.measurement.values, tmin.measurement.values, prcp.measurement.values]).T
    df = pd.DataFrame(arr / 10.0, index=tmin.date, columns=['TMAX', 'TMIN', 'PRCP'])
    if year is not None:
        start_date = datetime.datetime(year, 1, 1)
        end_date = datetime.datetime(year, 12, 31)
        df = df[(df.index >= start_date) & (df.index <= end_date)]
    df['days'] = (df.index - df.index.min()).days
    return df

def plot_regression(model: BayesianLinearRegression, data_train: pd.DataFrame, data_test: pd.DataFrame, N_train: int, year: int) -> Tuple[plt.Figure, BayesianLinearRegression]:
    x_train = data_train.days.values[:N_train, np.newaxis] * 1.0
    y_train = data_train.TMAX.values[:N_train, np.newaxis]
    
    Phi_train = expand(x_train)
    model.fit(Phi_train, y_train, verbose=True)
    
    x_days = np.arange(366)[:, np.newaxis]
    Phi_days = expand(x_days)
    y_days_pred, y_days_var = model.predict(Phi_days)
    
    x_test = data_test.days.values[:, np.newaxis] * 1.0
    Phi_test = expand(x_test)
    y_test = data_test.TMAX.values[:, np.newaxis]
    y_test_pred, y_test_var = model.predict(Phi_test)
    
    mse_train = np.mean((y_train - model.predict(Phi_train)[0])**2)
    mse_test = np.mean((y_test - y_test_pred)**2)
    
    print(f"Training MSE: {mse_train:.4f}")
    print(f"Test MSE: {mse_test:.4f}")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x_train, y_train, '.', label=f"Train MSE = {mse_train:.2f}")
    ax.plot(x_test, y_test, '.', label=f"Test MSE = {mse_test:.2f}")
    ax.plot(x_days, y_days_pred, 'r-', label="Prediction")
    ax.fill_between(x_days.flatten(), 
                    y_days_pred.flatten() - 2*np.sqrt(y_days_var.flatten()), 
                    y_days_pred.flatten() + 2*np.sqrt(y_days_var.flatten()), 
                    alpha=0.2, color='r')
    ax.set_ylim([-27, 39])
    ax.set_xlabel("Day of the year")
    ax.set_ylabel("Maximum Temperature - degree C")
    ax.set_title(f"Year: {year}        N: {N_train}")
    ax.legend()
    
    return fig, model
