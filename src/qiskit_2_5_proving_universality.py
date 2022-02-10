"""
https://qiskit.org/textbook/ch-gates/proving-universality.html
"""

import warnings

from qiskit import QuantumCircuit

from utils import draw_quantum_circuit


def cnot_conjugation():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        q = QuantumCircuit(2)
        q.cnot(1, 0)
        q.x(1)
        q.cnot(1, 0)
        draw_quantum_circuit(q, draw_circuit=False, draw_bloch_sphere=False, draw_final_state=False)


if __name__ == '__main__':
    cnot_conjugation()
