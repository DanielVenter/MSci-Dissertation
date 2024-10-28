import pennylane as qml
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import math

np.random.seed(0)

AnalyticalMean = math.sinh(1) / math.exp(1)
print("Analytical Mean", AnalyticalMean)



def func_f(x):
    return np.sin(x) ** 2

numSamples: int = 1000
sampleData: list = np.random.randn(numSamples, 1)
values: list = func_f(sampleData)
MCMean: float = np.mean(values)

print("MC Mean", MCMean)

# # Create the figure and subplots
# fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# # Plot the histogram of sample points on the left subplot
# axs[0].hist(sampleData, bins=30, edgecolor='black', alpha=0.7)
# axs[0].set_title("Sample Points")
# axs[0].set_xlabel("Value")
# axs[0].set_ylabel("Frequency")

# # Plot the histogram of the function values on the right subplot
# axs[1].hist(values, bins=20, edgecolor='black', alpha=0.7)
# axs[1].axvline(MCMean, color='black', linestyle='--', label=f'MC Mean: {MCMean:.2f}')
# axs[1].text(MCMean, max(np.histogram(values, bins=20)[0]) * 0.8, 'MC Mean', rotation=90, verticalalignment='center')
# axs[1].set_title("Classic Monte Carlo\nMean of Function Values")
# axs[1].set_xlabel("Function Value")
# axs[1].set_ylabel("Frequency")
# axs[1].legend()

# # Show the plots
# plt.tight_layout()
# plt.show()

m = 5
M = 2 ** m

xmax = np.pi  # bound to region [-pi, pi]
xs = np.linspace(-xmax, xmax, M)

probs = np.array([norm().pdf(x) for x in xs])
probs /= np.sum(probs)

def func(i):
    return np.sin(xs[i]) ** 2

DiscreteMean = np.sum(func_f(xs) * probs)
print("Discrete Mean", DiscreteMean)

n = 10
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


qmc_probs = circuit()
phase_estimated = np.argmax(qmc_probs[:int(N / 2)]) / N
phase_estimated_value = (1 - np.cos(np.pi * phase_estimated)) / 2
print("Phase Estimated", phase_estimated_value)





theta_values = np.linspace(0, 1, len(qmc_probs)) # Convert to [0, 1] for normalized phase values
plt.figure(figsize=(8, 6))
plt.plot(theta_values, qmc_probs, label="Probability", color='blue')

# # Add vertical line for the estimated phase
plt.axvline(phase_estimated, color='black', linestyle='--', label=f'Estimated Phase: {phase_estimated:.2f}')
plt.text(phase_estimated, max(qmc_probs) * 0.8, 'Analytic Phase', rotation=90, verticalalignment='center')

# Labels and title
plt.title("Phase Estimation with QMC", fontsize = 20)
plt.xlabel(r"$\theta$", fontsize = 20)
plt.xlim(0.4, 0.6)
plt.ylabel("Probability" , fontsize = 20)
plt.legend()

# # Show the plot
plt.show()