from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UCRYGate
from qiskit_aer import AerSimulator

# Create a quantum circuit with 3 qubits
qc = QuantumCircuit(3)

# Prepare the control qubits in superposition
qc.h(0)
qc.h(1)

# Define rotation angles for the UCRY gate
angles = [0, 0.5, 1.0, 1.5]  # Replace with your desired angles

# Apply the UCRY gate: control qubits (0, 1), target qubit (2)
ucry_gate = UCRYGate(angles)
qc.append(ucry_gate, [0, 1, 2])

# Measure all qubits
qc.measure_all()

# Transpile the circuit for the simulator
simulator = AerSimulator()
transpiled_qc = transpile(qc, simulator)

# Execute the circuit on the Aer simulator
job = simulator.run(transpiled_qc, shots=1024)
result = job.result()

# Get the measurement counts
counts = result.get_counts()

# Calculate the probabilities for the target qubit (qubit 2)
prob_0 = sum(count for bitstring, count in counts.items() if bitstring[0] == '0') / 1024
prob_1 = sum(count for bitstring, count in counts.items() if bitstring[0] == '1') / 1024

# Print the probabilities
print(f"Probability of target qubit being |0>: {prob_0}")
print(f"Probability of target qubit being |1>: {prob_1}")

# Optional: Visualize the full histogram of measurement results
from qiskit.visualization import plot_histogram
plot_histogram(counts).show()
