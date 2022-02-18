"""
https://qiskit.org/textbook/ch-gates/phase-kickback.html
"""

from math import pi
from qiskit import QuantumCircuit

from qiskit_1_4_single_qubit_gates import state_x1
from utils import draw_quantum_circuit


def x_minus():
    qc = QuantumCircuit(1)
    qc.initialize(state_x1, 0)
    qc.x(0)
    draw_quantum_circuit(qc)


def superposition():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.x(1)
    qc.h(1)
    qc.cnot(0, 1)
    draw_quantum_circuit(qc)


def cnot_behavior():
    """
    Show that the following quantum circuits are equivalent:

         ┌───┐           ┌───┐     ┌───┐
    q_0: ┤ X ├      q_0: ┤ H ├──■──┤ H ├
         └─┬─┘           ├───┤┌─┴─┐├───┤
    q_1: ──■──      q_1: ┤ H ├┤ X ├┤ H ├
                         └───┘└───┘└───┘

    Unitary of both:
     1       0       0       0
     0       1       0       0
     0       0       0       1
     0       0       1       0
    """

    qc_1 = QuantumCircuit(2)
    qc_1.cx(1, 0)
    draw_quantum_circuit(qc_1, draw_final_state=0, draw_unitary=1)

    qc_2 = QuantumCircuit(2)
    qc_2.h(0)
    qc_2.h(1)
    qc_2.cx(0, 1)
    qc_2.h(0)
    qc_2.h(1)
    draw_quantum_circuit(qc_2, draw_final_state=0, draw_unitary=1)


def t_gate():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.x(1)
    # qc.cp(pi / 4, 0, 1)
    draw_quantum_circuit(qc, draw_bloch_sphere=1)


def quick_exercise_1():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cp(pi / 4, 0, 1)
    draw_quantum_circuit(qc, draw_bloch_sphere=1)


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
    # cnot_behavior()
    # t_gate()
    # quick_exercise_1()
    # quick_exercise_2()
    quick_exercise_3()
