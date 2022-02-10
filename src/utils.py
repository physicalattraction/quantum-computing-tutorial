"""
https://qiskit.org/textbook/ch-gates/phase-kickback.html
"""

from enum import Enum
from typing import List

from math import pi, sqrt
from qiskit import Aer, QiskitError, QuantumCircuit, execute
from qiskit.visualization import plot_bloch_multivector, plot_histogram


class State(Enum):
    """
    The definitions are non-normalized for clarity
    """

    z0 = '0'  # (1, 0)
    z1 = '1'  # (0, 1)
    x0 = '+'  # (1, 1)
    x1 = '-'  # (1, -1)
    y0 = '↺'  # (1, -i)
    y1 = '↻'  # (1, i)


SIMPLE_STATES = (State.z0, State.z1, State.x0, State.x1)


def normalize(state: List[float]) -> List[float]:
    """
    Given an non-normalized vector, return it normalized
    """

    norm_factor = sqrt(1 / sum(abs(elem) * abs(elem) for elem in state))
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
    """
    Helper function to round a complex number

    - Rounds both real and imaginary components to two decimals
    - Removes real and imaginary parts when they're 0
    """

    if abs(c.real) > 1E-3 and abs(c.imag) > 1E-3:
        return f'{c.real:^5.2f} + {c.imag:>5.2f} i'
    elif abs(c.real) > 1E-3:
        return f'{c.real:>5.2f}'
    elif abs(c.imag) > 1E-3:
        return f'{c.imag:>5.2f} j'
    else:
        return f'{0:>5}'


def draw_quantum_circuit(qc: QuantumCircuit, draw_circuit=True,
                         draw_unitary=True, draw_final_state=True,
                         draw_bloch_sphere=True, draw_histogram=False):
    if draw_circuit:
        # Visualize the quantum circuit
        print('Quantum circuit:')
        print(qc.draw())

    if draw_unitary:
        try:
            # Visualize the unitary operator
            backend = Aer.get_backend('unitary_simulator')
            unitary = execute(qc, backend).result().get_unitary()
            print('Unitary:')
            for row in unitary:
                print('  '.join([round_complex(elem) for elem in row]))
        except QiskitError:
            # If a qunatum circuit contains a measure operation, the process is
            # not reversible anymore, and hence cannot be represented by a
            # Unitary matrix. We just ignore this operation in that case.
            pass

    if draw_final_state or draw_bloch_sphere:
        # Visualize the final state
        # final_state for 2 qubits = a |00> + b |01> + c |10> + d |11>
        backend = Aer.get_backend('statevector_simulator')
        final_state = execute(qc, backend).result().get_statevector()

        if draw_final_state:
            print('Final state:')
            for elem in final_state:
                print(round_complex(elem))
        if draw_bloch_sphere:
            plot_bloch_multivector(final_state).show()

    if draw_histogram:
        backend = Aer.get_backend('statevector_simulator')
        results = execute(qc, backend).result().get_counts()
        plot_histogram(results).show()
