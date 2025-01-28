import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def conditional_default_prob(p0, rho, z):
    """Calculate conditional default probability given systematic factor z"""
    return norm.cdf((norm.ppf(p0) - np.sqrt(rho) * z) / np.sqrt(1 - rho))

def simulate_portfolio_losses(n_assets, p0s, rhos, lambdas, n_simulations):
    """Simulate portfolio losses using Monte Carlo with L = Σ λ_k X_k(Z)"""
    losses = np.zeros(n_simulations)
    
    for i in range(n_simulations):
        z = np.random.normal(0, 1)
        p_k = np.array([conditional_default_prob(p0s[k], rhos[k], z) 
                       for k in range(n_assets)])
        X_k = np.random.binomial(1, p_k)
        losses[i] = np.sum(lambdas * X_k)
    
    return losses

def calculate_var(losses, alpha):
    return np.percentile(losses, 100 * (1 - alpha))

def calculate_cvar(losses, alpha):
    var = calculate_var(losses, alpha)
    return np.mean(losses[losses >= var])

def calculate_risk_measures(losses, confidence_levels=[0.95, 0.99]):
    risk_measures = {
        'Expected Loss': np.mean(losses),
        'Unexpected Loss': np.std(losses),
        'VaR': {},
        'CVaR': {}
    }
    
    for alpha in confidence_levels:
        risk_measures['VaR'][alpha] = calculate_var(losses, alpha)
        risk_measures['CVaR'][alpha] = calculate_cvar(losses, alpha)
    
    return risk_measures

def plot_risk_measures(losses, alpha=0.95):
    """Plot loss distribution with VaR and CVaR visualization"""
    plt.figure(figsize=(12, 6))
    
    # Plot histogram
    plt.hist(losses, bins=50, density=True, alpha=0.6, label='Loss Distribution')
    
    # Calculate and plot VaR and CVaR
    var = calculate_var(losses, alpha)
    cvar = calculate_cvar(losses, alpha)
    
    print(var)
    print(cvar)
    
    plt.axvline(var, color='r', linestyle='--', 
                label=f'VaR_{alpha} = {var:.2f}')
    plt.axvline(cvar, color='g', linestyle='--', 
                label=f'CVaR_{alpha} = {cvar:.2f}')
    
    plt.title('Portfolio Loss Distribution with Risk Measures')
    plt.xlabel('Loss Amount')
    plt.ylabel('Density')
    plt.legend()
    return plt


n_assets = 5
p_zeros = [0.15, 0.25, 0.5, 0.12, 0.32]
rhos = [0.1, 0.05, 0.08, 0.03, 0.08]
lgd = [1, 1.5, 0.5, 1, 0.75]
# p0s = np.ones(n_assets) * 0.01
# rhos = np.ones(n_assets) * 0.2
# lambdas = np.random.uniform(0.5, 1.5, n_assets)

losses = simulate_portfolio_losses(n_assets, p_zeros, rhos, lgd, 10000)
risk_measures = calculate_risk_measures(losses)
plot_risk_measures(losses)