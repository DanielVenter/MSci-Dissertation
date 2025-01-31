import numpy as np
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import UCRYGate, Initialize
from qiskit_aer import Aer, AerSimulator
from qiskit.visualization import plot_histogram
from matplotlib import pyplot as plt
from scipy.stats import norm
from qiskit.quantum_info import Statevector

np.random.seed(0)

# Define the function and interval
def f(x):
    return np.sin(x) ** 2

# Discretize the interval
m = 5
M = 2 ** m
xs = np.linspace(-np.pi, np.pi, M)

probs = np.array([norm().pdf(x) for x in xs])
probs /= np.sum(probs)

amplitudes = np.sqrt(probs)

angles = 2 * np.asin(np.sqrt(f(xs)))



qc = QuantumCircuit(m+1, name="F")
init = Initialize(amplitudes)
qc.append(init, range(m))
init.label = "A"

# cr = ClassicalRegister(1, 'c')
# qc.add_register(cr)

ucry = UCRYGate(angles.tolist()).inverse()
ucry.label = 'R'
qc.append(ucry, [m] + list(range(m)))

# qc.measure(m,0)

F = qc.to_instruction(label="F")
F_i = F.inverse()


qc = QuantumCircuit(m+2)
qc.cz(m,m+1)
qc.append(F_i, [m] + list(range(m)))





qc.draw(output="mpl")
plt.show()


# simulator = AerSimulator()
# compiled_circuit = transpile(qc, simulator)
# job = simulator.run(compiled_circuit, shots=1024)
# result = job.result()
# counts = result.get_counts(qc)

