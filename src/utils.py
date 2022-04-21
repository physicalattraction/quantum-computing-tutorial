"""
https://qiskit.org/textbook/ch-gates/phase-kickback.html
"""
from enum import Enum
from fractions import Fraction
from typing import List, Optional, Sequence

import itertools
import math
from math import pi, sqrt
from qiskit import Aer, QiskitError, QuantumCircuit, execute, assemble
from qiskit.providers.aer.backends.compatibility import Operator, Statevector
from qiskit.visualization import plot_bloch_multivector, plot_histogram, plot_state_qsphere


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


STATES = (State.z0, State.z1, State.x0, State.x1, State.y0, State.y1)
SIMPLE_STATES = (State.z0, State.z1, State.x0, State.x1)

STATE_TO_VECTOR_LOOKUP = {
    # State.z0: np.array([1, 0], dtype=np.complex128),
    # State.z1: np.array([0, 1], dtype=np.complex128),
    # State.x0: np.array([1 / sqrt(2), 1 / sqrt(2)], dtype=np.complex128),
    # State.x1: np.array([1 / sqrt(2), -1 / sqrt(2)], dtype=np.complex128),
    # State.y0: np.array([1 / sqrt(2), -1j / sqrt(2)], dtype=np.complex128),
    # State.y1: np.array([1 / sqrt(2), 1j / sqrt(2)], dtype=np.complex128),
    State.z0: [1, 0],
    State.z1: [0, 1],
    State.x0: [1 / sqrt(2), 1 / sqrt(2)],
    State.x1: [1 / sqrt(2), -1 / sqrt(2)],
    State.y0: [1 / sqrt(2), -1j / sqrt(2)],
    State.y1: [1 / sqrt(2), 1j / sqrt(2)],
}


def states_to_vector(*states: State) -> List[complex]:
    """
    Return the final state of the outer product of the various states


    >>> print_vector(states_to_vector(State.x1), transpose=True)
    [  1/√2  -1/√2  ]
    >>> print_vector(states_to_vector(State.y0), transpose=True)
    [  1/√2  -1/√2j  ]
    >>> print_vector(states_to_vector(State.z0, State.z1), transpose=True)
    [  0  1  0  0  ]
    >>> print_vector(states_to_vector(State.z0, State.z1, State.x0), transpose=True)
    [  0  0  1/√2  1/√2  0  0  0  0  ]
    >>> print_vector(states_to_vector(State.z0, State.x0, State.y1), transpose=True)
    [  1/2  1/2j  1/2  1/2j  0  0  0  0  ]
    """

    vectors = [STATE_TO_VECTOR_LOOKUP[state] for state in states]
    return [math.prod(combination) for combination in itertools.product(*vectors)]


def vector_to_states(vector: Sequence[complex]) -> Optional[str]:
    """
    Write the given final state into a product of single states

    If this is not possible with only simple states, return None

    >>> vector_to_states([1/sqrt(2), -1/sqrt(2)])
    '| - >'
    >>> vector_to_states([1/sqrt(2), -1j/sqrt(2)])
    '| ↺ >'
    >>> vector_to_states([0, 1, 0, 0])
    '| 0 1 >'
    >>> vector_to_states([0, 0, 1/sqrt(2), 1/sqrt(2), 0, 0, 0, 0])
    '| 0 1 + >'
    """

    if not isinstance(vector, list):
        # The input vector can e.g. be a qiskit Statevector
        vector = list(vector)
    nr_qubits = int(math.log2(len(vector)))
    for states in itertools.product(STATES, repeat=nr_qubits):
        if vector == states_to_vector(*states):
            return '| ' + ' '.join([state.value for state in states]) + ' >'
    return None


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


def display_float(number: float) -> str:
    """
    Helper function to display floats as square roots whenever possible

    Note: all outputs are right justified to 5 characters

    >>> display_float(0)
    '     0'
    >>> display_float(1)
    '     1'
    >>> display_float(2)
    '     2'
    >>> display_float(3)
    '     3'
    >>> display_float(-3)
    '    -3'
    >>> display_float(1/2)
    '   1/2'
    >>> display_float(1/3)
    '   1/3'
    >>> display_float(2/3)
    '   2/3'
    >>> display_float(-2/3)
    '  -2/3'
    >>> display_float(sqrt(2))
    '    √2'
    >>> display_float(1/sqrt(2))
    '  1/√2'
    >>> display_float(sqrt(4/3))
    '  2/√3'
    >>> display_float(-sqrt(4/3))
    ' -2/√3'
    >>> display_float(0.866)
    '  0.87'
    >>> display_float(-0.866)
    ' -0.87'
    """

    threshold = 1E-5
    number_is_negative = number < -threshold
    number = abs(number)

    def _float_as_integer(_number: float) -> int:
        possible_result = int(round(_number))
        if abs(_number - possible_result) < threshold:
            return possible_result

    def _float_as_fraction(_number: float) -> Fraction:
        possible_result = Fraction(_number).limit_denominator(64)
        if abs(_number - possible_result) / _number < threshold:
            return possible_result

    def _float_as_sqrt(_number: float) -> str:
        if root := _float_as_integer(_number * _number):
            return f'√{root}'
        if fraction := _float_as_fraction(_number * _number):
            numerator = fraction.numerator
            denominator = fraction.denominator
            if simplified_numerator := _float_as_integer(sqrt(numerator)):
                numerator = simplified_numerator
            else:
                numerator = f'√{numerator}'
            if simplified_denominator := _float_as_integer(sqrt(denominator)):
                denominator = simplified_denominator
            else:
                denominator = f'√{denominator}'
            return f'{numerator}/{denominator}'

    def _float_as_rounded_float(_number: float) -> str:
        return f'{_number:.2f}'

    for f in [_float_as_integer, _float_as_fraction,
              _float_as_sqrt, _float_as_rounded_float]:
        result = f(number)
        if result is not None:
            break

    if number == 0:
        number_is_negative = False
    result = str(result)
    if number_is_negative:
        result = '-' + result
    return result.rjust(6)


def print_vector(vector: List[complex], transpose=False):
    """
    Helper function to print a vector
    """

    elems = [display_complex(elem) for elem in vector]
    if transpose:
        elems = ['['] + [elem.strip() for elem in elems] + [']']
        print('  '.join(elems))
    else:
        print('\n'.join(elems))


def print_matrix(matrix: Sequence[List[complex]]):
    """
    Helper function to print a matrix
    """

    for row in matrix:
        print('  '.join([display_complex(elem) for elem in row]))


def display_complex(c: complex) -> str:
    """
    Helper function to round a complex number

    - Displays real and imaginary components as fractions or sqrts when possible
    - Rounds these components to two decimals otherwise
    - Removes real and imaginary parts when they're 0

    >>> display_complex(0)
    '     0'
    >>> display_complex(1/2)
    '   1/2'
    >>> display_complex(1j/2)
    '   1/2j'
    >>> display_complex(1/2+1j/2)
    '   1/2 +    1/2j'
    >>> display_complex(-sqrt(2)-sqrt(4/3)*1j)
    '   -√2 +  -2/√3j'
    """

    threshold = 1E-5

    if abs(c.real) > threshold and abs(c.imag) > threshold:
        return f'{display_float(c.real)} + {display_float(c.imag)}j'
    elif abs(c.imag) > threshold:
        return display_float(c.imag) + 'j'
    else:
        return display_float(c.real)


def draw_quantum_circuit(qc: QuantumCircuit, draw_circuit=True,
                         draw_unitary=False, draw_final_state=True,
                         draw_bloch_sphere=False, draw_q_sphere=False,
                         draw_histogram=False, draw_simulate=False,
                         use_row_vector=False):
    if draw_circuit:
        # Visualize the quantum circuit
        print('Quantum circuit:')
        print(qc.draw())

    if draw_unitary:
        try:
            # Visualize the unitary operator
            unitary = get_unitary(qc)
            print('Unitary:')
            print_matrix(unitary)
        except QiskitError:
            # If a qunatum circuit contains a measure operation, the process is
            # not reversible anymore, and hence cannot be represented by a
            # Unitary matrix. We just ignore this operation in that case.
            pass

    if draw_final_state or draw_bloch_sphere or draw_q_sphere:
        # Visualize the final state
        # final_state for 2 qubits = a |00> + b |01> + c |10> + d |11>
        final_state = get_final_state(qc)

        if draw_final_state:
            print('Final state:')
            if final_state_as_simple_states := vector_to_states(final_state):
                print(final_state_as_simple_states)
            else:
                print_vector(final_state, transpose=use_row_vector)
        if draw_bloch_sphere:
            plot_bloch_multivector(final_state).show()
        if draw_q_sphere:
            plot_state_qsphere(final_state).show()

    if draw_histogram:
        backend = Aer.get_backend('statevector_simulator')
        results = execute(qc, backend).result().get_counts()
        print('Histogram: ')
        for key, value in results.items():
            print(f'{key}: {value}')
        plot_histogram(results).show()

    if draw_simulate:
        aer_sim = Aer.get_backend('aer_simulator')
        shots = 1024
        qobj = assemble(qc, shots=shots)
        results = aer_sim.run(qobj).result()
        counts = results.get_counts()
        plot_histogram(counts).show()
        for key, value in counts.items():
            print(f'{key}: {value}')
        return counts


def get_final_state(qc: QuantumCircuit) -> Statevector:
    backend = Aer.get_backend('statevector_simulator')
    final_state = execute(qc, backend).result().get_statevector()
    return final_state


def get_unitary(qc: QuantumCircuit) -> Operator:
    """
    Get the unitary representing the given quantum circuit
    """

    backend = Aer.get_backend('unitary_simulator')
    unitary = execute(qc, backend).result().get_unitary()
    return unitary


def count_calls(func):
    """
    Add a variable `count` to a function that keeps track of how often it is called

    Usage:
    @count_calls
    def f(x, y):
        return x*y

    for a in range(3):
        for b in range(4):
            f(a, b)
    assert f.count == 12
    """

    def call_counter(*args, **kwargs):
        call_counter.count += 1
        return func(*args, **kwargs)

    call_counter.count = 0
    return call_counter
