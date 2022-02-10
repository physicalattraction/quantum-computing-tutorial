from math import pi
from qiskit import QuantumCircuit

from utils import SIMPLE_STATES, State, bit_to, draw_quantum_circuit

choices = SIMPLE_STATES


def single_states():
    """
    Print the definitions of the single state eigenvectors
    """

    for state in choices:
        qc = QuantumCircuit(1)
        print(f'|{state.value}>')
        bit_to(qc, 0, state)
        draw_quantum_circuit(qc, draw_circuit=False, draw_unitary=False, draw_bloch_sphere=False)
        print('----')


def dual_states():
    """
    Print the definitions of the combinations of all single-state eigenvectors
    """

    for state_0 in choices:
        for state_1 in choices:
            print(f'|{state_1.value}{state_0.value}>')
            qc = QuantumCircuit(2)
            bit_to(qc, 0, state_0)
            bit_to(qc, 1, state_1)
            draw_quantum_circuit(qc, draw_circuit=False, draw_unitary=False, draw_bloch_sphere=False)
            print('----')


def x():
    for state in choices:
        print(f'X |{state.value}>')
        qc = QuantumCircuit(1)
        bit_to(qc, 0, state)
        qc.x(0)
        draw_quantum_circuit(qc, draw_circuit=False, draw_unitary=False, draw_bloch_sphere=False)
        print('----')


def h():
    for state in choices:
        print(f'H |{state.value}>')
        qc = QuantumCircuit(1)
        bit_to(qc, 0, state)
        qc.h(0)
        draw_quantum_circuit(qc, draw_circuit=False, draw_unitary=False, draw_bloch_sphere=False)
        print('----')


def cnot():
    for state_0 in choices:
        for state_1 in choices:
            print(f'CNOT |{state_1.value}{state_0.value}>')
            qc = QuantumCircuit(2)
            bit_to(qc, 0, state_0)
            bit_to(qc, 1, state_1)
            qc.cnot(0, 1)
            draw_quantum_circuit(qc, draw_circuit=False, draw_unitary=False, draw_bloch_sphere=False)
            print('----')


def t():
    """
    The T-gate is the controlled U gate with
        ( u_00 u_01 )   ( 0      0     )
    U = ( u_10 u_11 ) = ( 0 exp(i pi /4)
    """

    for state_0 in choices:
        for state_1 in choices:
            print(f'T |{state_0.value}{state_1.value}>')
            qc = QuantumCircuit(2)
            bit_to(qc, 0, state_0)
            bit_to(qc, 1, state_1)
            qc.cu1(pi / 4, 1, 0)
            draw_quantum_circuit(qc, draw_circuit=False, draw_unitary=True, draw_bloch_sphere=False)
            print('----')


if __name__ == '__main__':
    # single_states()
    # dual_states()
    # x()
    # h()
    cnot()
    # tt()
