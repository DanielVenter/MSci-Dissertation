import pennylane as qml
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import math
import pandas as pd

true_mean = math.sinh(1) / math.exp(1)

def func_f(x):
    return np.sin(x) ** 2

def abs_error(m1,m2):
    return abs(m1 - m2)

def relative_error(m1,m2):
    return abs_error(m1,m2)/abs(m1)

def mse(m1,m2):
    return (m2 - m1) ** 2

def print_data(name, mean, true_mean, samples):
    return({'Name': name, 'Mean': mean, 'Absolute Error': abs_error(true_mean, mean), 'Relative Error': relative_error(true_mean, mean), 'MSE': mse(true_mean, mean), 'Samples': samples})


m = 5
M = 2 ** m

xmax = np.pi  # bound to region [-pi, pi]
xs = np.linspace(-xmax, xmax, M)

probs = np.array([norm().pdf(x) for x in xs])
probs /= np.sum(probs)

def func(i):
    return np.sin(xs[i]) ** 2

mse_values = []
num_samples = []x
data = []
for n in range(1, 6):
    num_samples.append(n)
    N = 2 ** n

    target_wires = range(m + 1)
    estimation_wires = range(m + 1, n + m + 1)

    dev = qml.device("default.qubit", wires=(n + m + 1))

    @qml.qnode(dev)
    def circuit():
        qml.templates.QuantumMonteCarlo(
            probs,
            func,
            target_wires=target_wires,
            estimation_wires=estimation_wires,
        )
        return qml.probs(estimation_wires)

    results = []
    for _ in range(5):
        qmc_probs = circuit()
        phase_estimated = np.argmax(circuit()[:int(N / 2)]) / N
        answer = (1 - np.cos(np.pi * phase_estimated)) / 2
        results.append(answer)
    variance = np.var(results)
    
    format_data = print_data("QMC",answer,true_mean, n )
    format_data["Variance"] = variance
    
    mse_values.append(format_data.get("MSE"))
    data.append(format_data)
    
df = pd.DataFrame(data) 
print(df.to_latex(index=False, float_format="%.4f"))



plt.figure(figsize=(10, 6))
plt.plot(num_samples, mse_values, marker='o', linestyle='-', color='b')
# plt.xscale('log')  # Set x-axis to logarithmic scale
plt.xlabel("Number of Samples")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("MSE vs Number of Samples")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()