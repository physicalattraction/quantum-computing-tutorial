"""
https://qiskit.org/textbook/ch-gates/phase-kickback.html
"""

from enum import Enum
from fractions import Fraction
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

    number_is_negative = number < 0
    number = abs(number)

    threshold = 1E-5

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

    result = str(result)
    if number_is_negative:
        result = '-' + result
    return result.rjust(6)


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
                print('  '.join([display_complex(elem) for elem in row]))
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
                print(display_complex(elem))
        if draw_bloch_sphere:
            plot_bloch_multivector(final_state).show()

    if draw_histogram:
        backend = Aer.get_backend('statevector_simulator')
        results = execute(qc, backend).result().get_counts()
        plot_histogram(results).show()
