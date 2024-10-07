"""
Monte Carlo model to simulate the growth 
of an investment portfolio over time.
"""
import numpy as np
from helpers import (
    get_random_returns, 
    get_confidence_levels,
    get_yearly_percentiles)

from plots import (
    plot_histogram,
    plot_yearly_percentiles)


def perform_simulation(inputs: dict):
    """
    Performs a simulation to find out how much
    the pot is worth in £ after years of growth.
    
    Returns:
        pot (float)     - the final amount at the end 
        history (list)  - the yearly history of results 
                          [10000, 11000, 12000, ...]
        
    """
    years = inputs['end_age'] - inputs['start_age']
    pot = inputs['starting_pot']
    returns = get_random_returns(years=years) 
    mean_return = (np.mean(returns) - 1) * 100
    
    history = []
    
    for i in range(years):
        annual_return = returns[i]
        pot *= annual_return
        pot += inputs['annual_contributions']
        history.append(int(pot))
        
    return pot, history, mean_return
    
    
def perform_monte_carlo(inputs: dict, n: int = 1000):
    pot_sizes = []
    results = []
    mean_returns = []
    
    for i in range(n):
        final_amount, history, mean_return = perform_simulation(inputs)
        pot_sizes.append(final_amount)
        results.append(history)
        mean_returns.append(mean_return)
        
    lower_confidence, upper_confidence = get_confidence_levels(pot_sizes)
    
    print('Monte carlo model done :)', end='\n')
    print('Plots saved to /outputs folder')
    print('Mean return across all simulations: ', end='')
    print(f'{round(np.mean(mean_returns), 1)}%')
    
    return {
        'pot_sizes': pot_sizes,
        'results': results, 
        'yearly_percentiles': get_yearly_percentiles(results, inputs),
        'lower_confidence': lower_confidence,
        'upper_confidence': upper_confidence,
        'mean_returns': mean_returns
    }
    
    
if __name__ == "__main__":
    inputs = {
        'start_age': 20,
        'end_age': 65,
        'starting_pot': 5000,
        'annual_contributions': 500 * 12, 
        'target_amount': 300000,
        'n_simulations': 10000
    }
    
    mc = perform_monte_carlo(inputs, 
                             n=inputs['n_simulations'])
    
    plot_histogram(mc['pot_sizes'], 
                   mc['upper_confidence'], 
                   mc['lower_confidence'])
    
    plot_yearly_percentiles(inputs=inputs,
                            df=mc['yearly_percentiles'])
