"""
https://qiskit.org/textbook/ch-algorithms/quantum-phase-estimation.html
"""

from typing import Dict, Callable, Optional, List

from math import pi
from qiskit import QuantumCircuit

from qiskit_3_5_qft import invqft
from utils import draw_quantum_circuit, bit_to, State


def _get_estimate_phase_for_gate(apply_gate: Callable, name: str, initial_states: List[State], nr_counting_bits: int):
    nr_u_bits = len(initial_states)

    # Initialize the quantum circuit in the given initial (multiqubit) state
    qc = QuantumCircuit(nr_counting_bits + nr_u_bits, nr_counting_bits)
    for index, initial_state in enumerate(initial_states, start=nr_counting_bits):
        bit_to(qc, index, initial_state)

    # Next, we apply Hadamard gates to the counting qubits:
    for i in range(nr_counting_bits):
        qc.h(i)

    # Do the controlled-U operations:
    repetitions = 1
    for counting_qubit in range(nr_counting_bits):
        for i in range(repetitions):
            apply_gate(qc, counting_qubit)  # This is CU
        repetitions *= 2

    # We apply the inverse quantum Fourier transformation to convert the state of the counting register.
    qc.barrier()
    invqft(qc, nr_counting_bits)

    # Measure
    qc.barrier()
    for n in range(nr_counting_bits):
        qc.measure(n, n)
    qc.barrier()

    counts = draw_quantum_circuit(qc, draw_circuit=False, draw_simulate=True)
    estimate = get_estimate_from_counts(counts)

    initial_state_str = '|' + ''.join([s.value for s in initial_states]) + '>'
    if estimate is not None:
        print(f'{name} {initial_state_str} = exp(2i π θ) {initial_state_str} with θ = {estimate}')
    else:
        print(f'{initial_state_str} is not an eigenvalue of {name}')


def estimate_phase_of_t_gate(initial_state: State, nr_counting_bits: int = 6):
    def apply_ct_gate(qc: QuantumCircuit, counting_qubit: int):
        qc.cp(pi / 4, counting_qubit, nr_counting_bits)  # This is CU

    _get_estimate_phase_for_gate(apply_ct_gate, 'T', [initial_state], nr_counting_bits)


def estimate_phase_of_r_gate(initial_state: State, nr_counting_bits: int = 6):
    angle = 0.17

    def apply_r_gate(qc: QuantumCircuit, counting_qubit: int):
        qc.cp(2 * pi * angle, counting_qubit, nr_counting_bits)  # This is CU

    _get_estimate_phase_for_gate(apply_r_gate, f'R({angle})', [initial_state], nr_counting_bits)


def estimate_phase_of_z_gate(initial_state: State, nr_counting_bits: int = 6):
    def apply_z_gate(qc: QuantumCircuit, counting_qubit: int):
        qc.cz(counting_qubit, nr_counting_bits)  # This is CU

    _get_estimate_phase_for_gate(apply_z_gate, f'Z', [initial_state], nr_counting_bits)


def estimate_phase_of_x_gate(initial_state: State, nr_counting_bits: int = 6):
    def apply_x_gate(qc: QuantumCircuit, counting_qubit: int):
        qc.cx(counting_qubit, nr_counting_bits)  # This is CU

    _get_estimate_phase_for_gate(apply_x_gate, f'X', [initial_state], nr_counting_bits)


def estimate_phase_of_cnot_gate(initial_state: List[State], nr_counting_bits: int = 6):
    def apply_cnot_gate(qc: QuantumCircuit, counting_qubit: int):
        qc.ccx(counting_qubit, nr_counting_bits, nr_counting_bits + 1)  # This is CU

    _get_estimate_phase_for_gate(apply_cnot_gate, f'CNOT', initial_state, nr_counting_bits)


def estimate_phase_of_cs_gate(initial_state: List[State], nr_counting_bits: int = 6):
    def apply_cnot_gate(qc: QuantumCircuit, counting_qubit: int):
        qc.cc(counting_qubit, nr_counting_bits, nr_counting_bits + 1)  # This is CU

    _get_estimate_phase_for_gate(apply_cnot_gate, f'CNOT', initial_state, nr_counting_bits)


def get_estimate_from_counts(counts: Dict[str, int], print_output=False) -> Optional[str]:
    """
    Get the estimate for θ for the two most likely values

    :param counts: Counts from a QC simulation or real job
    :returns: If a state has more than 60% of the counts, return that state, otherwise return None
    """

    first_count = second_count = 0
    first_key = second_key = ''
    for key, value in counts.items():
        if value > first_count:
            second_count = first_count
            second_key = first_key
            first_count = value
            first_key = key
        elif value > second_count:
            second_count = value
            second_key = key
    first_answer = int(first_key, base=2) / 2 ** (len(first_key))
    estimate = sum([int(key, base=2) * value for key, value in counts.items()]) / \
               (2 ** len(first_key) * sum(counts.values()))

    if first_count < 0.6 * sum(counts.values()):
        if print_output:
            print('The initial state was probably not an eigenstate')
        return None

    if print_output:
        print(f'{first_key=}, {second_key}')
        if second_key:
            second_answer = int(second_key, base=2) / 2 ** (len(first_key))
            print(f'The estimate is between {first_answer} and {second_answer}')
        else:
            print(f'The estimate is {first_answer:.4f}')

        print(f'The estimate based on all counts is {estimate:.4f}')

    return first_answer


if __name__ == '__main__':
    # 1 bit gates
    estimate_phase_of_t_gate(initial_state=State.z0)
    estimate_phase_of_t_gate(initial_state=State.z1)
    estimate_phase_of_r_gate(initial_state=State.z0)
    estimate_phase_of_r_gate(initial_state=State.z1)
    estimate_phase_of_z_gate(initial_state=State.z0)
    estimate_phase_of_z_gate(initial_state=State.z1)
    estimate_phase_of_x_gate(initial_state=State.z0)
    estimate_phase_of_x_gate(initial_state=State.z1)
    estimate_phase_of_x_gate(initial_state=State.x0)
    estimate_phase_of_x_gate(initial_state=State.x1)

    # 2 bit gates
    estimate_phase_of_cnot_gate(initial_state=[State.z1, State.x0])
    estimate_phase_of_cnot_gate(initial_state=[State.z1, State.x1])
