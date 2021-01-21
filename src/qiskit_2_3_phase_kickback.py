"""
https://qiskit.org/textbook/ch-gates/phase-kickback.html
"""
from enum import Enum
from typing import List

from math import pi, sqrt
from qiskit import Aer, QuantumCircuit, execute
from qiskit.visualization import plot_bloch_multivector

from qiskit_1_4_single_qubit_gates import state_x1


class State(Enum):
    z0 = '0'
    z1 = '1'
    x0 = '+'
    x1 = '-'
    y0 = '↺'
    y1 = '↻'


def normalize(state: List[float]) -> List[float]:
    norm_factor = sqrt(1 / sum(elem * elem for elem in state))
    return [norm_factor * elem for elem in state]


def bit_to(qc: QuantumCircuit, bit: int, state: State):
    """
    Assuming that bit 0 is in state |0>, bring it into the given state
    """

    if state == State.z0:
        pass
    elif state == State.z1:
        qc.x(bit)
    elif state == State.x0:
        qc.ry(pi / 2, bit)
    elif state == State.x1:
        qc.ry(-pi / 2, bit)
    elif state == State.y0:
        qc.rx(pi / 2, bit)
    elif state == State.y1:
        qc.rx(-pi / 2, bit)


def round_complex(c: complex) -> str:
    if abs(c.real) > 1E-3 and abs(c.imag) > 1E-3:
        return f'{c.real:^5.2f} + {c.imag:>5.2f} i'
    elif abs(c.real) > 1E-3:
        return f'{c.real:>5.2f}'
    elif abs(c.imag) > 1E-3:
        return f'{c.imag:>5.2f} j'
    else:
        return f'{0:>5}'


def draw_quantum_circuit(qc: QuantumCircuit):
    # Visualize thew quantum circuit
    print(qc.draw())

    # Visualize the unitary operator
    backend = Aer.get_backend('unitary_simulator')
    unitary = execute(qc, backend).result().get_unitary()
    print('Unitary:')
    for row in unitary:
        print('  '.join([round_complex(elem) for elem in row]))

    # Visualize the final state
    # final_state for 2 qubits = a |00> + b |01> + c |10> + d |11>
    backend = Aer.get_backend('statevector_simulator')
    final_state = execute(qc, backend).result().get_statevector()
    print('Final state:')
    for elem in final_state:
        print(round_complex(elem))
    plot_bloch_multivector(final_state).show()


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


if __name__ == '__main__':
    # x_minus()
    # superposition()
    t_gate()
