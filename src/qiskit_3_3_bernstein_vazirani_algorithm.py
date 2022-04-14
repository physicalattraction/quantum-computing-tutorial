"""
https://qiskit.org/textbook/ch-algorithms/bernstein-vazirani.html
"""

import numpy as np
from qiskit import IBMQ, QuantumCircuit, transpile
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor
from qiskit.visualization import plot_histogram

from settings.local import get_secret
from utils import draw_quantum_circuit

BitString = str


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


def get_secret_s(n: int) -> BitString:
    """
    Return a random bit string of length n, e.g. 10001100
    """

    return ''.join([str(np.random.randint(2)) for _ in range(n)])


@count_calls
def f(input_string: BitString, secret_string: BitString) -> int:
    """
    Apply the function f with the secret bit string s

    In words, f counts the number of bits that are 1 in both the input string and the secret string,
    and returns that count modulo 2

    >>> f('1', '1')
    1
    >>> f('1', '0')
    0
    >>> f('100', '000')
    0
    >>> f('100', '100')
    1
    >>> f('100', '010')
    0
    >>> f('111', '011')
    0
    """

    result = 0
    for bit_input, bit_s in zip(input_string, secret_string):
        result += int(bit_input) * int(bit_s)
    return result % 2


def classical_solution(s: str):
    """
    Find the secret string by calling f 8 times
    """

    n = len(s)
    result = ''
    for i in range(n):
        input_string = ''.join(['1' if j == i else '0' for j in range(8)])
        result += str(f(input_string, s))
    assert s == result, f'Deduced secret is {result}, expected it to be {s}'
    assert n == f.count, f'Function f is called {f.count} times, expected it to be called {n} times'


def get_quantum_circuit(s: str) -> QuantumCircuit:
    n = len(s)

    # We need a circuit with n qubits, plus one auxiliary qubit
    # Also need n classical bits to write the output to
    qc = QuantumCircuit(n + 1, n)

    # put auxiliary in state |->
    qc.h(n)
    qc.z(n)

    # Apply Hadamard gates before querying the oracle
    for i in range(n):
        qc.h(i)

    # Apply barrier
    qc.barrier()

    # Apply the inner-product oracle
    s = s[::-1]  # reverse s to fit qiskit's qubit ordering
    for q in range(n):
        if s[q] == '0':
            qc.i(q)
        else:
            qc.cx(q, n)

    # Apply barrier
    qc.barrier()

    # Apply Hadamard gates after querying the oracle
    for i in range(n):
        qc.h(i)

    # Measurement
    for i in range(n):
        qc.measure(i, i)

    return qc


def quantum_solution_simulated(s: str):
    qc = get_quantum_circuit(s)
    draw_quantum_circuit(qc, draw_final_state=False, draw_histogram=True)


def quantum_solution_real(s: str):
    n = len(s)
    qc = get_quantum_circuit(s)

    # Load our saved IBMQ accounts and get the least busy backend device with less than or equal to 5 qubits
    IBMQ.save_account(token=get_secret('IBM_TOKEN'), overwrite=True)
    IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q')

    def backend_is_suitable(x):
        return n + 1 <= x.configuration().n_qubits <= n + 4 and \
               not x.configuration().simulator and \
               x.status().operational == True

    backend = least_busy(provider.backends(filters=backend_is_suitable))
    print(f'Least busy {backend=}')

    # Run our circuit on the least busy backend. Monitor the execution of the job in the queue
    shots = 1024
    transpiled_bv_circuit = transpile(qc, backend)
    job = backend.run(transpiled_bv_circuit, shots=shots)
    job_monitor(job, interval=2)

    # Get the results from the computation
    results = job.result()
    answer = results.get_counts()

    print('Histogram: ')
    for key, value in answer.items():
        print(f'{key}: {value}')
    plot_histogram(answer).show()


if __name__ == '__main__':
    s = get_secret_s(4)
    print(f'Secret to guess: {s}')
    # classical_solution(s)
    # quantum_solution_simulated(s)
    quantum_solution_real(s)
