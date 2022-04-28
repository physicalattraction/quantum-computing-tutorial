"""
https://qiskit.org/textbook/ch-algorithms/quantum-fourier-transform.html
"""

from math import pi, log, ceil
from qiskit import QuantumCircuit

from utils import draw_quantum_circuit, get_final_state, compare_vectors, print_histogram_from_real


def _qft_rotations(circuit, n=None) -> QuantumCircuit:
    """
    Performs qft on the first n qubits in circuit (without swaps)
    """

    if n is None:
        n = circuit.num_qubits
    if n == 0:
        return circuit
    n -= 1
    circuit.h(n)
    for qubit in range(n):
        circuit.cp(pi / 2 ** (n - qubit), qubit, n)
    _qft_rotations(circuit, n)


def _invqft_rotations(circuit, nr_qubits: int = None, n=None) -> QuantumCircuit:
    """
    Performs inverse qft on the first n qubits in circuit (without swaps)
    """

    if nr_qubits is None:
        nr_qubits = circuit.num_qubits
    if n is None:
        n = 0
    if n == nr_qubits:
        return circuit
    for qubit in reversed(range(n)):
        circuit.cp(-pi / 2 ** (n - qubit), n, qubit)
    circuit.h(n)
    _invqft_rotations(circuit, nr_qubits, n + 1)


def _swap_registers(circuit: QuantumCircuit, n: int = None) -> QuantumCircuit:
    if n is None:
        n = circuit.num_qubits
    for qubit in range(n // 2):
        circuit.swap(qubit, n - qubit - 1)
    return circuit


def qft(circuit: QuantumCircuit, n: int = None) -> QuantumCircuit:
    """
    Add an inverse QFT circuit to the existing circuit

    :param circuit: Circuit to add the QFT to
    :param n: Apply to the first n bits. If not given, use all qubits in the circuit
    :return: Resulting circuit (although it is also edited in place, it is returned to enable chaining)
    """

    _qft_rotations(circuit, n)
    _swap_registers(circuit, n)
    return circuit


def invqft(circuit: QuantumCircuit, n: int = None) -> QuantumCircuit:
    """
    Add an inverse QFT circuit to the existing circuit

    :param circuit: Circuit to add the Inv QFT to
    :param n: Apply to the first n bits. If not given, use all qubits in the circuit
    :return: Resulting circuit (although it is also edited in place, it is returned to enable chaining)
    """

    _swap_registers(circuit, n)
    _invqft_rotations(circuit, n)
    return circuit


def get_initial_qc(n: int, number: int) -> QuantumCircuit:
    """
    Get a quantum circuit in an initial state `number`,
    e.g. if number = 5, the quantum circuit is in the initial state |101>
    """

    circuit = QuantumCircuit(n)
    for index, bit in enumerate(bin(number)[2:]):
        if bit == '1':
            circuit.x(n-1-index)
    return circuit


def get_initial_inverse_qc(n: int, number: int) -> QuantumCircuit:
    """
    Get a quantum circuit in an initial state `~number`,
    e.g. if number = 5, the quantum circuit is in the initial state QFT(|101>)
    """

    circuit = QuantumCircuit(n)
    for qubit in range(n):
        circuit.h(qubit)
    for i in range(n):
        circuit.p(number * 2 * pi * 2 ** (i - n), i)
    return circuit


if __name__ == '__main__':
    initial_number = 27
    min_qubits = int(ceil(log(initial_number, 2)))

    # Get QC in initial state
    print('\nOriginal\n')
    qc = get_initial_qc(min_qubits, initial_number)
    original_state = get_final_state(qc)
    draw_quantum_circuit(qc, draw_circuit=False, draw_final_state=True, draw_bloch_sphere=True)

    # Quantum Fourier transform it
    print('\nQFT\n')
    qft(qc)
    draw_quantum_circuit(qc, draw_bloch_sphere=True)

    # Verify the output
    print('\nOriginal inv\n')
    inv_qc = get_initial_inverse_qc(min_qubits, initial_number)
    draw_quantum_circuit(inv_qc, draw_bloch_sphere=True)
    compare_vectors(get_final_state(qc), get_final_state(inv_qc))

    # Quantum inverse Fourier transform it
    print('\nQFT inv\n')
    invqft(inv_qc)
    compare_vectors(get_final_state(inv_qc), original_state)

    # Measure all qubits
    inv_qc.measure_all()
    counts = draw_quantum_circuit(inv_qc, draw_simulate=True)

    # Verify that the output is the original input
    assert len(counts) == 1
    key = set(counts.keys()).pop()
    assert int(key, 2) == initial_number, f'{key} ({int(key, 2)}) != {initial_number}'

    print_histogram_from_real(inv_qc, nr_shots=2048)
