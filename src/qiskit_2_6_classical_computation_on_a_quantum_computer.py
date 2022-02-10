"""
https://qiskit.org/textbook/ch-gates/oracles.html
"""

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    from qiskit import QuantumCircuit, QuantumRegister

from utils import draw_quantum_circuit


def take_out_garbage():
    def draw(_qc):
        draw_quantum_circuit(_qc, draw_histogram=False, draw_bloch_sphere=False, draw_unitary=True,
                             draw_final_state=True)

    input_bit = QuantumRegister(1, 'input')
    output_bit = QuantumRegister(1, 'output')
    garbage_bit = QuantumRegister(1, 'garbage')

    # noinspection PyPep8Naming
    # Uf = QuantumCircuit(input_bit, output_bit, garbage_bit)
    # Uf.cx(input_bit[0], output_bit[0])
    # draw(Uf)

    # noinspection PyPep8Naming
    Vf = QuantumCircuit(input_bit, output_bit, garbage_bit)
    Vf.h(input_bit[0])
    Vf.cx(input_bit[0], garbage_bit[0])
    Vf.cx(input_bit[0], output_bit[0])
    draw(Vf)

    # qc = Uf + Vf.inverse()
    # draw(qc)

    final_output_bit = QuantumRegister(1, 'final-output')
    copy = QuantumCircuit(output_bit, final_output_bit)
    copy.cx(output_bit, final_output_bit)
    draw(copy)

    draw(Vf.inverse() + copy + Vf)


if __name__ == '__main__':
    take_out_garbage()
