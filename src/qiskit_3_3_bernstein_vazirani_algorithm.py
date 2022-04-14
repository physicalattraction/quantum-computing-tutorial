"""
https://qiskit.org/textbook/ch-algorithms/deutsch-jozsa.html
"""

import numpy as np
from qiskit import Aer, IBMQ, QuantumCircuit, execute
from qiskit.circuit import Gate
from qiskit.providers import BaseBackend
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor
from qiskit.visualization import plot_histogram
from qiskit_textbook import problems

from utils import draw_quantum_circuit

BitString = str


def get_secret_s(n: int) -> BitString:
    """
    Return a random bit string of length n, e.g. 10001100
    """

    return ''.join([str(np.random.randint(2)) for _ in range(n)])


def f(input_string: BitString, secret_string: BitString) -> int:
    """
    Apply the function f with the secret bit string s

    >>> f('1', '1')
    0
    >>> f('1', '0')
    1
    >>> f('100', '000')
    1
    >>> f('101', '010')
    1
    """

    result = 0
    for bit_input, bit_s in zip(input_string, secret_string):
        result += int(bit_input) * int(bit_s)
    return result % 2


def classical_solution():
    """
    Find the secret string by calling f 8 times
    """

    n = 8
    s = get_secret_s(n)
    result = ''
    for i in range(n):
        input_string = ''.join(['1' if j == i else '0' for j in range(8)])
        result += str(f(input_string, s))
    assert(s == result)


if __name__ == '__main__':
    # print(get_secret_s(8))
    classical_solution()
