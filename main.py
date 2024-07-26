# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 16:40:20 2024

@author: fahad
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from load_plot import load_temperature_data, plot_regression
from bayes_linear_regression import BayesianLinearRegression

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(2)

    # Load and prepare data
    year = 1900
    filename = 'GM000003342.csv'  # You may need to change this to the correct filename
    df = load_temperature_data(filename, year=year)
    idx = np.random.permutation(df.shape[0])
    idx_train, idx_test = idx[:100], idx[100:]
    data_train, data_test = df.iloc[idx_train], df.iloc[idx_test]

    # Initialize and fit the model
    model = BayesianLinearRegression(alpha_0=1e-6, beta_0=1e-6)
    N_train = len(data_train)
    fig, fitted_model = plot_regression(model, data_train, data_test, N_train, year)
    plt.show()

    # Print additional information about the fitted model
    print(f'Optimized Alpha: {fitted_model.alpha}, Beta: {fitted_model.beta}')
    print(f'Posterior Mean:\n{fitted_model.posterior_mean}')
    print(f'Posterior Covariance:\n{fitted_model.posterior_covariance}')
