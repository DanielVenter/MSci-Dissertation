import pennylane as qml
from pennylane.templates import QuantumPhaseEstimation
from pennylane import numpy as np

phase = 2
target_wires = [0]

n_estimation_wires = 5
estimation_wires = range(len(target_wires), n_estimation_wires + 1)

dev = qml.device("default.qubit", wires=n_estimation_wires + 1)

@qml.qnode(dev)
def circuit():

    for i, wire in enumerate(target_wires):

        # used to set the state of the target_wires to 0.5 probability
        qml.Hadamard(wires=wire)
        
        unitary = qml.RX(phase, wires=i).matrix()
        QuantumPhaseEstimation(
            unitary,
            target_wires=wire,
            estimation_wires=estimation_wires,
        )

    return qml.probs(estimation_wires)

phase_estimated = np.argmax(circuit()) / 2 ** n_estimation_wires

# Need to rescale phase due to convention of RX gate
phase_estimated = 4 * np.pi * (1 - phase_estimated)
print(phase_estimated)