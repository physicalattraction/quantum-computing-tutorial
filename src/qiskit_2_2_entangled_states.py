"""
https://qiskit.org/textbook/ch-gates/multiple-qubits-entangled-states.html
"""

from math import sqrt

from qiskit import *
from qiskit.visualization import plot_histogram


def draw_quantum_circuit():
    qc = QuantumCircuit(3)
    # Apply H-gate to each qubit:
    for qubit in range(3):
        qc.h(qubit)
    # See the circuit:
    print(qc.draw())

    # Let's see the result
    backend = Aer.get_backend('statevector_simulator')
    final_state = execute(qc, backend).result().get_statevector()
    print(final_state)


def single_qubit_gates_on_multi_qubit_vectors():
    qc = QuantumCircuit(3)
    qc.x(0)
    qc.z(1)
    qc.h(2)

    backend = Aer.get_backend('unitary_simulator')
    unitary = execute(qc, backend).result().get_unitary()
    for row in unitary:
        print(row * sqrt(2))


def first_cnot():
    qc = QuantumCircuit(2)

    # Apply H-gate to the first:
    qc.h(0)
    qc.x(1)
    qc.cx(0, 1)
    print(qc.draw())
    backend = Aer.get_backend('statevector_simulator')
    final_state = execute(qc, backend).result().get_statevector()
    # print(final_state)

    results = execute(qc, backend).result().get_counts()
    plot_histogram(results).show()

    backend = Aer.get_backend('unitary_simulator')
    unitary = execute(qc, backend).result().get_unitary()
    for row in unitary:
        print([round(abs(elem * sqrt(2))) for elem in row])


if __name__ == '__main__':
    # draw_quantum_circuit()
    # single_qubit_gates_on_multi_qubit_vectors()
    first_cnot()
