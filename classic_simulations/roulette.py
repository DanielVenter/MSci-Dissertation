import random

def play_roulette():
    total_slots = 37
    red_probability   = (18 / total_slots) * 100
    black_probability = (18 / total_slots) * 100
    green_probability = (1 / total_slots) * 100
    
    possible_outcomes = ["red", "black", "green"]
    probabilities = [red_probability, black_probability, green_probability]
    
    outcome = random.choices(
        possible_outcomes,
        weights=probabilities,
        k=1
    )[0]
        
    return outcome

def perform_simulation(n_times=1000, choice="red"):   
    results_local = { "red": 0, "black": 0, "green": 0 }
    
    for _ in range(n_times):
        outcome = play_roulette()
        results_local[outcome] += 1
           
    win_percentage_local = results_local[choice] / n_times
        
    return results_local, win_percentage_local

        
if __name__ == "__main__":
    results, win_percentage = perform_simulation(n_times=1000000, 
                                                 choice="red")
    
    print(results)
    print(win_percentage)
