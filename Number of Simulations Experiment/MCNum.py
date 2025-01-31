import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd


true_mean = math.sinh(1) / math.exp(1)

def func_f(x):
    return np.sin(x) ** 2

def mc_mean(numSamples):
    sampleData: list = np.random.randn(numSamples, 1)
    values: list = func_f(sampleData)
    MCMean: float = np.mean(values)
    return print_data("MC Mean", MCMean, true_mean, numSamples)
    
    
def abs_error(m1,m2):
    return abs(m1 - m2)

def relative_error(m1,m2):
    return abs_error(m1,m2)/abs(m1)

def mse(m1,m2):
    return (m2 - m1) ** 2

def print_data(name, mean, true_mean, samples):
    return({'Name': name, 'Mean': mean, 'Absolute Error': abs_error(true_mean, mean), 'Relative Error': relative_error(true_mean, mean), 'MSE': mse(true_mean, mean), 'Samples': samples})

mse_values = []
num_samples = []
data = []
variances = []

for i in range(5):
    samples = 10 ** i
    num_samples.append(samples)
    answer = mc_mean(samples)
    
    results = []
    for _ in range(5):
        result = mc_mean(samples)
        results.append(result['Mean']) 
    variance = np.var(results)
       
    answer["Variance"] = variance
    
    data.append(answer)
    mse_values.append(answer.get('mse'))




df = pd.DataFrame(data) 
print(df.to_latex(index=False, float_format="%.4f"))



# plt.figure(figsize=(10, 6))
# plt.plot(num_samples, mse_values, marker='o', linestyle='-', color='b')
# plt.xscale('log')  # Set x-axis to logarithmic scale
# plt.xlabel("Number of Samples (log scale)")
# plt.ylabel("Mean Squared Error (MSE)")
# plt.title("MSE vs Number of Samples (Logarithmic Scale)")
# plt.grid(True, which="both", linestyle="--", linewidth=0.5)
# plt.show()
