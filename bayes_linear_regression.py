# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 16:38:24 2024

@author: fahad
"""

import numpy as np
from typing import Tuple, Optional

class BayesianLinearRegression:
    """
    Implements Bayesian Linear Regression using the evidence approximation method.

    This class provides a framework for performing Bayesian inference in linear regression settings,
    optimizing the hyperparameters alpha (precision of the weight prior) and beta (precision of the noise)
    through evidence approximation. The method iteratively updates these hyperparameters to maximize
    the marginal likelihood (evidence) of the observed data.

    Attributes:
        alpha_0 (float): Initial value for the precision of the weight prior distribution.
        beta_0 (float): Initial value for the precision of the noise.
        max_iter (int): Maximum number of iterations for the hyperparameter optimization.
        rtol (float): Relative tolerance for convergence of the hyperparameters.
        alpha (float): Optimized precision of the weight prior distribution.
        beta (float): Optimized precision of the noise.
        posterior_mean (np.ndarray): Mean of the posterior distribution of the weights.
        posterior_covariance (np.ndarray): Covariance matrix of the posterior distribution of the weights.
    """

    def __init__(self, alpha_0: float = 1e-5, beta_0: float = 1e-5, max_iter: int = 200, rtol: float = 1e-5):
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        self.max_iter = max_iter
        self.rtol = rtol
        self.alpha = None
        self.beta = None
        self.posterior_mean = None
        self.posterior_covariance = None

    def fit(self, Phi: np.ndarray, t: np.ndarray, verbose: bool = False) -> Tuple[float, float, np.ndarray, np.ndarray]:
        N, M = Phi.shape
        eigenvalues_0 = np.linalg.eigvalsh(Phi.T.dot(Phi))
        beta = self.beta_0
        alpha = self.alpha_0

        for i in range(self.max_iter):
            beta_prev, alpha_prev = beta, alpha
            eigenvalues = eigenvalues_0 * beta
            m_N, S_N, S_N_inv = self.posterior(Phi, t, alpha, beta, return_inverse=True)

            gamma = np.sum(eigenvalues / (eigenvalues + alpha))
            alpha = gamma / np.sum(m_N ** 2)
            beta_inv = 1 / (N - gamma) * np.sum((t - Phi.dot(m_N)) ** 2)
            beta = 1 / beta_inv

            if np.isclose(alpha_prev, alpha, rtol=self.rtol) and np.isclose(beta_prev, beta, rtol=self.rtol):
                if verbose:
                    print(f'Convergence after {i + 1} iterations.')
                break

        self.alpha, self.beta = alpha, beta
        self.posterior_mean, self.posterior_covariance = m_N, S_N

        if verbose:
            print(f'Optimized Alpha: {alpha}')
            print(f'Optimized Beta: {beta}')

        return alpha, beta, m_N, S_N

    def posterior(self, Phi: np.ndarray, t: np.ndarray, alpha: float, beta: float, return_inverse: bool = False) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        S_N_inv = alpha * np.eye(Phi.shape[1]) + beta * Phi.T.dot(Phi)
        S_N = np.linalg.inv(S_N_inv)
        m_N = beta * S_N.dot(Phi.T).dot(t)

        return (m_N, S_N, S_N_inv) if return_inverse else (m_N, S_N)

    def predict(self, Phi_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        y = Phi_test.dot(self.posterior_mean)
        y_var = 1 / self.beta + np.sum(Phi_test.dot(self.posterior_covariance) * Phi_test, axis=1)
        return y, y_var

def identity_basis_function(x: np.ndarray) -> np.ndarray:
    return x

def expand(x: np.ndarray, bf=identity_basis_function) -> np.ndarray:
    return np.column_stack([np.ones(x.shape[0]), bf(x)])
