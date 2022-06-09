import os.path
import pickle
from enum import Enum
from fractions import Fraction
from typing import List, Union
from typing import Optional, Sequence

import itertools
import math
from math import pi, sqrt
from qiskit import IBMQ, QuantumCircuit, transpile, assemble, Aer
from qiskit import QiskitError, execute
from qiskit.providers.aer.backends.compatibility import Operator, Statevector
from qiskit.providers.backend import BackendV1 as Backend
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor
from qiskit.visualization import plot_bloch_multivector, plot_state_qsphere
from qiskit.visualization import plot_histogram

from settings.local import get_secret


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
SRC_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(SRC_DIR)
CIRCUIT_DIR = os.path.join(ROOT_DIR, 'circuits')


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


def print_vector(vector: Union[Statevector, List[complex]], transpose=False):
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
                         draw_unitary=False, draw_final_state=False,
                         draw_bloch_sphere=False, draw_q_sphere=False,
                         draw_histogram=False, draw_simulate=False,
                         nr_shots=1024, use_row_vector=False):
    if draw_simulate:
        aer_sim = Aer.get_backend('aer_simulator')
        qc = load_transpiled_quantum_circuit(qc, aer_sim)

    if draw_circuit:
        # Visualize the quantum circuit
        print(f'\n\nQuantum circuit "{qc.name}":')
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
        shots = nr_shots
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


def print_histogram_from_real(qc: QuantumCircuit, nr_shots: int = 1024):
    n = qc.num_qubits
    # if n > 5:
    #     print(f'There are no free quantum computers available with more than 5 qubits, '
    #           f'you requested one with {n} qubits.')
    #     return

    # Load our saved IBMQ accounts and get the least busy backend device with less than or equal to 5 qubits
    IBMQ.save_account(token=get_secret('IBM_TOKEN'), overwrite=True)
    IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q')

    def backend_is_suitable(x: Backend):
        return n <= x.configuration().n_qubits and \
               not x.configuration().simulator and \
               x.status().operational

    print(f'Fetching list of suitable backends with at least {n} qubits...')
    backend: Backend = least_busy(provider.backends(filters=backend_is_suitable))
    print(f'Least busy {backend=}')

    transpiled_bv_circuit = load_transpiled_quantum_circuit(qc, backend, optimization_level=3)

    # Run our circuit on the least busy backend. Monitor the execution of the job in the queue
    job = backend.run(transpiled_bv_circuit, shots=nr_shots)
    job_monitor(job, interval=2)

    # Get the results from the computation
    results = job.result()
    answer = results.get_counts()

    print('Histogram: ')
    for key, value in answer.items():
        print(f'{key}: {value}')
    plot_histogram(answer).show()
    return answer


def compare_vectors(x: Statevector, y: Statevector):
    if x != y:
        print_vector(x, transpose=True)
        print('is not equal to')
        print_vector(y, transpose=True)


def save_quantum_circuit(qc: QuantumCircuit, filename: str):
    """
    Pickle the quantum circuit for later use

    Useful for when you have performed a costly optimized transpilation
    """

    if not filename.endswith('.circuit'):
        filename += '.circuit'
    with open(os.path.join(CIRCUIT_DIR, filename), 'wb') as f:
        pickle.dump(qc, f)


def load_transpiled_quantum_circuit(qc: QuantumCircuit, backend: Backend, optimization_level: int = 1) -> QuantumCircuit:
    """
    Return the transpiled quantum circuit for the given quantum circuit.

    We first check if the given quantum circuit has been transpiled already
    If so, we load and unpickle it. If not, we transpile it now

    This lookip is based on the name of the Quantum circuit. If you don't name your circuit
    explicitly, the circuit gets a name like circuit-2. Since this could contain anything,
    we do not load transpiled circuits for those Quantum circuits.

    :param qc: any QuantumCircuit
    :param backend: Backend to transpile the circuit on
    :return: transpiled QuantumCircuit
    """

    filename = f'{backend.name()}||{qc.name}.circuit'

    if not qc.name.startswith('circuit'):
        if circuit := load_quantum_circuit(filename):
            print(f'Reuse transpiled circuit {filename}')
            return circuit

    print(f'Transpiling circuit {filename}')
    qc = transpile(qc, backend, optimization_level=optimization_level)
    save_quantum_circuit(qc, filename)
    return qc


def load_quantum_circuit(filename: str) -> Optional[QuantumCircuit]:
    """
    Unpickle the quantum circuit saved earlier

    If the file does not exist, return None

    Useful for when you have performed a costly optimized transpilation
    """

    if not filename.endswith('.circuit'):
        filename += '.circuit'
    filepath = os.path.join(CIRCUIT_DIR, filename)

    if not os.path.isfile(filepath):
        return None

    with open(filepath, 'rb') as f:
        return pickle.load(f)
