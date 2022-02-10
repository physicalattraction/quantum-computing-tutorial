"""
https://qiskit.org/textbook/ch-gates/phase-kickback.html
"""

from math import pi
from qiskit import QuantumCircuit

from qiskit_1_4_single_qubit_gates import state_x1
from utils import State, bit_to, draw_quantum_circuit


def x_minus():
    qc = QuantumCircuit(1)
    qc.initialize(state_x1, 0)
    qc.x(0)
    draw_quantum_circuit(qc)


def superposition():
    qc = QuantumCircuit(2)
    bit_to(qc, 0, State.x1)
    bit_to(qc, 1, State.z1)
    qc.cnot(1, 0)
    # qc.cnot(0, 1)
    draw_quantum_circuit(qc)


def t_gate():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.x(1)
    qc.cu1(pi / 4, 0, 1)
    draw_quantum_circuit(qc)


def quick_exercise_1():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cu1(pi / 4, 0, 1)
    draw_quantum_circuit(qc)


def quick_exercise_2():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.x(1)
    qc.cu1(-pi / 2, 0, 1)
    draw_quantum_circuit(qc)


def quick_exercise_3():
    qc = QuantumCircuit(2)
    qc.x(0)
    qc.x(1)
    qc.cu1(pi / 4, 0, 1)
    draw_quantum_circuit(qc)


if __name__ == '__main__':
    # x_minus()
    # superposition()
    # t_gate()
    # quick_exercise_1()
    # quick_exercise_2()
    quick_exercise_3()
