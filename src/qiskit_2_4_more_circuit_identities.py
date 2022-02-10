"""
https://qiskit.org/textbook/ch-gates/more-circuit-identities.html
"""

from math import pi
from qiskit import QuantumCircuit
from qiskit.circuit import Gate

from utils import draw_quantum_circuit


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
    alpha = 1  # arbitrarily define alpha to allow drawing of circuit

    qc = QuantumCircuit(2)
    qc.append(C, [1])
    qc.cz(0, 1)
    qc.append(B, [1])
    qc.cz(0, 1)
    qc.append(A, [1])
    qc.p(alpha, 0)
    print(qc.draw())


def toffoli():
    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)
    draw_quantum_circuit(qc, draw_final_state=False, draw_bloch_sphere=False)


def build_controlled_rotation():
    qc = QuantumCircuit(3)
    theta = pi / 4  # chosen arbitrarily
    qc.cp(theta / 2, 1, 2)
    qc.cx(0, 1)
    qc.cp(-theta / 2, 1, 2)
    qc.cx(0, 1)
    qc.cp(theta / 2, 0, 2)
    draw_quantum_circuit(qc, draw_final_state=False, draw_bloch_sphere=False)


def single_cp():
    qc = QuantumCircuit(1)
    theta = pi / 4
    qc.p(theta, 0)
    draw_quantum_circuit(qc)

    qc = QuantumCircuit(2)
    theta = pi / 4
    qc.cp(theta, 0, 1)
    draw_quantum_circuit(qc)


def build_toffoli():
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
    draw_quantum_circuit(qc)

    qc = QuantumCircuit(3)
    qc.ch(0, 2)
    qc.cz(1, 2)
    qc.ch(0, 2)
    draw_quantum_circuit(qc)


def build_t_gates():
    # Rotate pi/4 along the Z-axis
    qc = QuantumCircuit(1)
    qc.t(0)
    draw_quantum_circuit(qc)

    # Rotate pi/4 along the X-axis
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.t(0)
    qc.h(0)
    draw_quantum_circuit(qc)

    # Rotate pi/4 along the X-axis, then around the Z-axis
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.t(0)
    qc.h(0)
    qc.t(0)
    draw_quantum_circuit(qc)


if __name__ == '__main__':
    # controlled_z()
    # controlled_y()
    # controlled_h()
    # swap_bits()
    # controlled_rotation_y()
    # controlled_rotation_generic()
    # build_controlled_rotation()
    # toffoli()
    # build_toffoli()
    # single_cp()
    build_t_gates()
