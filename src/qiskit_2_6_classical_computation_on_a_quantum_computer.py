"""
https://qiskit.org/textbook/ch-gates/oracles.html
"""

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    from qiskit import QuantumCircuit, QuantumRegister

from utils import draw_quantum_circuit, get_final_state, vector_to_states


def take_out_garbage():
    def draw(_qc):
        draw_quantum_circuit(_qc, draw_histogram=False, draw_bloch_sphere=False,
                             draw_unitary=False, draw_final_state=True)

    input_bit = QuantumRegister(1, 'input')
    output_bit = QuantumRegister(1, 'output')
    garbage_bit = QuantumRegister(1, 'garbage')
    final_output_bit = QuantumRegister(1, 'final-output')

    # noinspection PyPep8Naming
    Uf = QuantumCircuit(input_bit, output_bit, garbage_bit, final_output_bit)
    Uf.cx(input_bit[0], output_bit[0])
    # draw(Uf)

    # noinspection PyPep8Naming
    Vf = QuantumCircuit(input_bit, output_bit, garbage_bit, final_output_bit)
    Vf.cx(input_bit[0], garbage_bit[0])
    Vf.cx(input_bit[0], output_bit[0])
    # draw(Vf)

    qc = Uf.compose(Vf.inverse())
    # draw(qc)

    copy = QuantumCircuit(output_bit, final_output_bit)
    copy.cx(output_bit, final_output_bit)
    # draw(copy)

    # qc = Vf.inverse() + copy + Vf
    qc = Vf.inverse().compose(copy, qubits=[1, 3]).compose(Vf)
    draw(qc)


def quick_exercise_3_1():
    input_bit = QuantumRegister(1, 'input')
    output_bit = QuantumRegister(1, 'output')
    garbage_bit = QuantumRegister(1, 'garbage')
    final_output_bit = QuantumRegister(1, 'final-output')

    # noinspection PyPep8Naming
    Vf = QuantumCircuit(input_bit, output_bit, garbage_bit, final_output_bit)
    Vf.cx(input_bit[0], garbage_bit[0])
    Vf.cx(input_bit[0], output_bit[0])

    copy = QuantumCircuit(output_bit, final_output_bit)
    copy.cx(output_bit, final_output_bit)

    # If output register is in |0>, the final_output will be a copy of the input
    qc = QuantumCircuit(input_bit, output_bit, garbage_bit, final_output_bit)
    assert_qc_is_in_final_state(qc, '| 0 0 0 0 >')
    qc = qc.compose(Vf.inverse()).compose(copy, qubits=[1, 3]).compose(Vf)
    assert_qc_is_in_final_state(qc, '| 0 0 0 0 >')  # final = input

    qc = QuantumCircuit(input_bit, output_bit, garbage_bit, final_output_bit)
    qc.x(0)
    assert_qc_is_in_final_state(qc, '| 0 0 0 1 >')
    qc = qc.compose(Vf.inverse()).compose(copy, qubits=[1, 3]).compose(Vf)
    assert_qc_is_in_final_state(qc, '| 1 0 0 1 >')  # final = input

    # If output register is in |1>, the final_output will not be a copy of the input
    qc = QuantumCircuit(input_bit, output_bit, garbage_bit, final_output_bit)
    qc.x(1)
    assert_qc_is_in_final_state(qc, '| 0 0 1 0 >')
    qc = qc.compose(Vf.inverse()).compose(copy, qubits=[1, 3]).compose(Vf)
    assert_qc_is_in_final_state(qc, '| 1 0 1 0 >')  # final != input

    qc = QuantumCircuit(input_bit, output_bit, garbage_bit, final_output_bit)
    qc.x(1)
    qc.x(0)
    assert_qc_is_in_final_state(qc, '| 0 0 1 1 >')
    qc = qc.compose(Vf.inverse()).compose(copy, qubits=[1, 3]).compose(Vf)
    assert_qc_is_in_final_state(qc, '| 0 0 1 1 >')  # final  != input


def assert_qc_is_in_final_state(qc: QuantumCircuit, expected_final_state: str):
    final_statevector = get_final_state(qc)
    final_state = vector_to_states(final_statevector)
    if expected_final_state != final_state:
        print(f'QC is in {final_state} instead of  in {expected_final_state}')


if __name__ == '__main__':
    take_out_garbage()
    # quick_exercise_3_1()
