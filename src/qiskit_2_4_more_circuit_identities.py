"""
https://qiskit.org/textbook/ch-gates/more-circuit-identities.html
"""

from math import pi
from qiskit import Aer, QuantumCircuit, execute
from qiskit.circuit import Gate

from utils import State, bit_to, draw_quantum_circuit, print_matrix


def unitaries():
    qc = QuantumCircuit(2)
    qc.h(1)
    print('H')
    draw_quantum_circuit(qc, draw_unitary=True, draw_final_state=False)

    qc = QuantumCircuit(2)
    qc.cz(0, 1)
    print('CZ')
    draw_quantum_circuit(qc, draw_unitary=True, draw_final_state=False)

    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    print('CX')
    draw_quantum_circuit(qc, draw_unitary=True, draw_final_state=False)

    qc = QuantumCircuit(2)
    print('H CX H')
    qc.h(1)
    qc.cx(0, 1)
    qc.h(1)
    draw_quantum_circuit(qc, draw_unitary=True, draw_final_state=False)


def controlled_z():
    qc = QuantumCircuit(2)
    qc.cz(0, 1)
    draw_quantum_circuit(qc, draw_final_state=False, draw_bloch_sphere=False)

    qc = QuantumCircuit(2)
    qc.h(1)
    qc.cnot(0, 1)
    qc.h(1)
    draw_quantum_circuit(qc, draw_final_state=False, draw_bloch_sphere=False)


def controlled_y():
    qc = QuantumCircuit(2)
    qc.cy(0, 1)
    draw_quantum_circuit(qc, draw_final_state=False, draw_bloch_sphere=False)

    qc = QuantumCircuit(2)
    qc.sdg(1)
    qc.cnot(0, 1)
    qc.s(1)
    draw_quantum_circuit(qc, draw_final_state=False, draw_bloch_sphere=False)


def controlled_h():
    qc = QuantumCircuit(2)
    qc.ch(0, 1)
    draw_quantum_circuit(qc, draw_final_state=False, draw_bloch_sphere=False)

    qc = QuantumCircuit(2)
    qc.ry(pi / 4, 1)
    qc.cnot(0, 1)
    qc.ry(-pi / 4, 1)
    draw_quantum_circuit(qc, draw_final_state=False, draw_bloch_sphere=False)


def swap_bits():
    qc = QuantumCircuit(2)
    qc.swap(0, 1)
    draw_quantum_circuit(qc, draw_final_state=False, draw_bloch_sphere=False)

    qc = QuantumCircuit(2)
    qc.cnot(0, 1)
    qc.cnot(1, 0)
    qc.cnot(0, 1)
    draw_quantum_circuit(qc, draw_final_state=False, draw_bloch_sphere=False)


def swap_bits_x():
    qc = QuantumCircuit(2)
    bit_to(qc, 0, State.x0)
    bit_to(qc, 1, State.x1)
    draw_quantum_circuit(qc, draw_final_state=True, draw_unitary=False)
    qc.cnot(0, 1)
    qc.cnot(1, 0)
    qc.cnot(0, 1)
    draw_quantum_circuit(qc, draw_final_state=True, draw_unitary=False)

    qc = QuantumCircuit(2)
    bit_to(qc, 1, State.x0)
    bit_to(qc, 0, State.x1)
    draw_quantum_circuit(qc, draw_final_state=True, draw_unitary=False)


def controlled_rotation_y():
    qc = QuantumCircuit(2)
    theta = pi  # theta can be anything (pi chosen arbitrarily)
    qc.ry(theta / 2, 1)
    qc.cx(0, 1)
    qc.ry(-theta / 2, 1)
    qc.cx(0, 1)
    draw_quantum_circuit(qc, draw_final_state=False, draw_bloch_sphere=False)


def controlled_rotation_generic():
    A = Gate('A', 1, [])
    B = Gate('B', 1, [])
    C = Gate('C', 1, [])
    alpha = pi / 4  # arbitrarily define alpha to allow drawing of circuit

    qc = QuantumCircuit(2)
    qc.append(C, [1])
    qc.cz(0, 1)
    qc.append(B, [1])
    qc.cz(0, 1)
    qc.append(A, [1])
    qc.p(alpha, 1)
    draw_quantum_circuit(qc, draw_final_state=False)


def toffoli():
    qc = QuantumCircuit(3)
    qc.x(0)
    qc.x(1)
    # qc.x(2)
    draw_quantum_circuit(qc, draw_circuit=False, draw_unitary=False)
    qc.ccx(0, 1, 2)
    draw_quantum_circuit(qc, draw_circuit=False, draw_unitary=False)


def build_controlled_rotation():
    qc = QuantumCircuit(3)
    theta = pi  # chosen arbitrarily
    qc.cp(theta / 2, 1, 2)
    qc.cx(0, 1)
    qc.cp(-theta / 2, 1, 2)
    qc.cx(0, 1)
    qc.cp(theta / 2, 0, 2)
    draw_quantum_circuit(qc, draw_final_state=False, draw_bloch_sphere=False)


def single_cp():
    qc = QuantumCircuit(2)
    theta = pi / 4
    qc.p(theta, 1)
    draw_quantum_circuit(qc)

    qc = QuantumCircuit(2)
    theta = pi / 4
    qc.cp(theta, 0, 1)
    draw_quantum_circuit(qc)


def build_toffoli():
    print('Decomposed Toffoli')
    qc = QuantumCircuit(3)
    qc.h(2)
    qc.cx(1, 2)
    qc.tdg(2)
    qc.cx(0, 2)
    qc.t(2)
    qc.cx(1, 2)
    qc.tdg(2)
    qc.cx(0, 2)
    qc.t(1)
    qc.t(2)
    qc.h(2)
    qc.cx(0, 1)
    qc.t(0)
    qc.tdg(1)
    qc.cx(0, 1)
    draw_quantum_circuit(qc, draw_final_state=False)

    print('CH CZ CH')
    qc = QuantumCircuit(3)
    # qc.ch(0, 2)
    qc.ry(pi / 4, 2)
    qc.cnot(0, 2)
    qc.ry(-pi / 4, 2)
    # qc.cz(1, 2)
    qc.h(2)
    qc.cx(1, 2)
    qc.h(2)
    # qc.ch(0, 2)
    qc.ry(pi / 4, 2)
    qc.cnot(0, 2)
    qc.ry(-pi / 4, 2)
    draw_quantum_circuit(qc, draw_final_state=False)

    print('CCX')
    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)
    draw_quantum_circuit(qc, draw_final_state=False)


def build_t_gates():
    # Rotate pi/4 along the Z-axis
    qc_1 = QuantumCircuit(1)
    qc_1.t(0)
    draw_quantum_circuit(qc_1, draw_final_state=False, draw_bloch_sphere=True)

    qc_2 = QuantumCircuit(1)
    qc_2.p(pi / 4, 0)
    backend = Aer.get_backend('unitary_simulator')
    unitary_1 = execute(qc_1, backend).result().get_unitary()
    unitary_2 = execute(qc_2, backend).result().get_unitary()
    assert (unitary_1 == unitary_2), f'{unitary_1}\n{unitary_2}'

    # Rotate pi/4 along the X-axis
    qc_3 = QuantumCircuit(1)
    qc_3.h(0)
    qc_3.t(0)
    qc_3.h(0)
    draw_quantum_circuit(qc_3, draw_final_state=True, draw_unitary=False, draw_bloch_sphere=True)

    qc_4 = QuantumCircuit(1)
    qc_4.p(pi / 2, 0)
    backend = Aer.get_backend('unitary_simulator')
    unitary_3 = execute(qc_3, backend).result().get_unitary()
    unitary_4 = execute(qc_4, backend).result().get_unitary()
    assert (unitary_3 == unitary_4)

    # Rotate pi/4 along the X-axis, then around the Z-axis
    qc_5 = QuantumCircuit(1)
    qc_5.h(0)
    qc_5.t(0)
    qc_5.h(0)
    qc_5.t(0)
    draw_quantum_circuit(qc_5, draw_final_state=False, draw_bloch_sphere=True)

    qc_6 = QuantumCircuit(1)
    qc_6.rx(pi / 2, 0)
    qc_6.rz(pi / 2, 0)
    backend = Aer.get_backend('unitary_simulator')
    unitary_5 = execute(qc_5, backend).result().get_unitary()
    unitary_6 = execute(qc_6, backend).result().get_unitary()
    assert (unitary_5 == unitary_6)


def keep_rotating():
    # Rotate pi/4 along the X-axis, then around the Z-axis
    qc = QuantumCircuit(1)
    for i in range(10):
        qc.h(0)
        qc.t(0)
        qc.h(0)
        qc.t(0)
        draw_quantum_circuit(qc, draw_circuit=False, draw_final_state=False, draw_bloch_sphere=True)


if __name__ == '__main__':
    # unitaries()
    # controlled_z()
    # controlled_y()
    # controlled_h()
    # swap_bits_x()
    # controlled_rotation_y()
    # controlled_rotation_generic()
    # toffoli()
    # single_cp()
    # build_controlled_rotation()
    # build_toffoli()
    build_t_gates()
    # keep_rotating()
