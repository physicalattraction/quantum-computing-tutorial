"""
https://qiskit.org/textbook/ch-states/atoms-computation.html
"""

from qiskit import QuantumCircuit

from utils import draw_quantum_circuit


def measure_qc():
    qc = QuantumCircuit(8)
    qc.x(7)
    draw_quantum_circuit(qc,
                         draw_circuit=True, draw_unitary=False,
                         draw_final_state=False, draw_bloch_sphere=True,
                         draw_histogram=True)


def half_adder():
    qc = QuantumCircuit(4, 2)
    qc.x(0)
    # qc.x(1)
    qc.barrier()
    # Set bit 2: 0 for 00 and 11, 1 for 10 and 01
    qc.cx(0, 2)
    qc.cx(1, 2)
    # Set bit 3: 0 for 00, 01 and 10, 1 for 11
    qc.ccx(0, 1, 3)
    qc.barrier()
    qc.measure(2, 0)
    qc.measure(3, 1)
    draw_quantum_circuit(qc,
                         draw_circuit=True, draw_unitary=False,
                         draw_final_state=False, draw_bloch_sphere=True,
                         draw_histogram=True)


if __name__ == '__main__':
    # measure_qc()
    half_adder()
