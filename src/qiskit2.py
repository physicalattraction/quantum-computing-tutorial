from typing import List

import qiskit as q
from math import sqrt
from qiskit.visualization import plot_bloch_multivector, plot_histogram

state_vector_simulator = q.Aer.get_backend('statevector_simulator')
qasm_simulator = q.Aer.get_backend('qasm_simulator')


def do_job(circuit):
    result = q.execute(circuit, backend=state_vector_simulator).result()
    statevec = result.get_statevector()

    nr_qubits = circuit.qregs[0].size
    nr_cbits = circuit.cregs[0].size
    circuit.measure(list(range(nr_qubits)), list(range(nr_cbits)))
    qasm_result = q.execute(circuit, backend=qasm_simulator, shots=2 ** 12).result()
    counts = qasm_result.get_counts()

    return statevec, counts


def normalize_vector(vector: List) -> List:
    norm = sqrt(sum([abs(x) ** 2 for x in vector]))
    return [x / norm for x in vector]


qc = q.QuantumCircuit(1, 1)  # Create a quantum circuit with one qubit
initial_state = normalize_vector([sqrt(1 / 3), sqrt(2 / 3)])
# initial_state = normalize_vector([sqrt(1 / 3), 1j * sqrt(2 / 3)])
qc.initialize(initial_state, 0)  # Apply initialisation operation to the 0th qubit
statevec, counts = do_job(qc)
print(statevec)
plot_bloch_multivector(statevec).show()
# plot_histogram([counts]).show()

qc.measure_all()
statevec, counts = do_job(qc)
print(statevec)
plot_bloch_multivector(statevec).show()