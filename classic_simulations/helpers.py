import random
import numpy as np
import pandas as pd


def get_random_returns(years: int):
    """
    Generates a list of random return percentages
    for the length of years required.
    """
    random_returns = []
    
    for i in range(years):
        high_negative_returns = (random.randint(-20, -8) / 1000) + 1
        low_negative_returns = (random.randint(-7, -1) / 1000) + 1
        low_returns = (random.randint(0, 4) / 100) + 1
        medium_returns = (random.randint(5, 9) / 100) + 1
        high_returns = (random.randint(10, 20) / 100) + 1
        
        possible_returns = [        # Weights
            high_negative_returns,  # 5  % chance
            low_negative_returns,   # 25 % chance
            low_returns,            # 40 % chance
            medium_returns,         # 25 % chance
            high_returns            # 5  % chance
        ]
        
        random_return = random.choices(
            possible_returns,
            weights=(5, 25, 40, 25, 5),
            k=1
        )[0]
        
        random_returns.append(
            random_return
        )
    
    return random_returns


def get_confidence_levels(pot_sizes):    
    upper_confidence = round(np.quantile(pot_sizes, 0.975), 2)
    lower_confidence = round(np.quantile(pot_sizes, 0.025), 2)
    
    return lower_confidence, upper_confidence


def get_yearly_percentiles(results, inputs) -> pd.DataFrame:
    """
    Finds the percentiles for each year.
    """
    results_rotated = list(zip(*results[::-1]))

    year = []
    age = []
    ninetieth_percentile = []
    seventy_fifth_percentile = []
    median = []
    twenty_fifth_percentile = []
    tenth_percentile = []
    
    for i, year_results in enumerate(results_rotated):
        new_age = (inputs['start_age'] + 1) + i
        ninetieth_percentile_value = np.percentile(year_results, 90)
        seventy_fifth_percentile_value = np.percentile(year_results, 75)
        median_value = np.median(year_results)
        twenty_fifth_percentile_value = np.percentile(year_results, 25)
        tenth_percentile_value = np.percentile(year_results, 10)
        
        year.append(i + 1)
        age.append(new_age)
        ninetieth_percentile.append(ninetieth_percentile_value)
        seventy_fifth_percentile.append(seventy_fifth_percentile_value)
        median.append(median_value)
        twenty_fifth_percentile.append(twenty_fifth_percentile_value)
        tenth_percentile.append(tenth_percentile_value)
        

    return pd.DataFrame(
        list(
            zip(year,
                age,
                ninetieth_percentile, 
                seventy_fifth_percentile,
                median, 
                twenty_fifth_percentile,
                tenth_percentile)
        ),
        columns=[
            'year',
            'age',
            '90th_percentile',
            '75th_percentile',
            'median', 
            '25th_percentile',
            '10th_percentile']
    )
    
