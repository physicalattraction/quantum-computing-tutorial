"""
https://qiskit.org/textbook/ch-gates/multiple-qubits-entangled-states.html
"""

from qiskit import *

from utils import draw_quantum_circuit


def single_qubit_gates_on_multi_qubit_vectors():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.x(1)
    qc.cx(0, 1)
    draw_quantum_circuit(qc, draw_histogram=1, draw_q_sphere=1)
    # draw_quantum_circuit(qc)


def xh():
    """
    Proof that X x H =
        0      0   0.71   0.71
        0      0   0.71  -0.71
     0.71   0.71      0      0
     0.71  -0.71      0      0
    And hence that X x H means X\q1> x H|q0>
    """

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.x(1)
    draw_quantum_circuit(qc)


def first_cnot():
    qc = QuantumCircuit(2)

    # Apply H-gate to the first:
    qc.h(0)
    # qc.x(1)
    qc.cx(0, 1)
    draw_quantum_circuit(qc, draw_histogram=True)


def exercise_2_1():
    qc = QuantumCircuit(3)
    qc.x(2)
    qc.z(1)
    qc.h(0)
    draw_quantum_circuit(qc)


def exercise_3_4():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.x(1)
    qc.cx(0, 1)
    draw_quantum_circuit(qc, draw_histogram=1, draw_q_sphere=1)


if __name__ == '__main__':
    # single_qubit_gates_on_multi_qubit_vectors()
    # xh()
    # first_cnot()
    exercise_2_1()
    # exercise_3_4()
