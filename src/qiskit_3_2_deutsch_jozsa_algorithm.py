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


def step_1():
    qc = QuantumCircuit(3)
    qc.x(0)
    qc.h(2)
    qc.h(1)
    qc.h(0)
    draw_quantum_circuit(qc, draw_bloch_sphere=False)


def make_oracle_balanced():
    qc = QuantumCircuit(4)
    qc.cnot(0, 1)
    qc.cnot(0, 2)
    qc.cnot(0, 3)
    draw_quantum_circuit(qc, draw_bloch_sphere=False)


def constant_oracle(draw=False):
    n = 3
    qc = QuantumCircuit(n + 1)

    # Place X-gates
    output = np.random.randint(2)
    if output == 1:
        qc.x(n)

    print('Output:', output)
    if draw:
        draw_quantum_circuit(qc, draw_bloch_sphere=False)

    return qc


def balanced_oracle(draw=False):
    n = 3
    qc = QuantumCircuit(n + 1)

    # Place X-gates
    bit_str = ''
    for i in range(n):
        if np.random.randint(2):
            bit_str += '1'
        else:
            bit_str += '0'
    for qubit, qubit_value in enumerate(bit_str):
        if qubit_value == '1':
            qc.x(qubit)

    qc.barrier()

    # Place CNOTs
    for qubit_value in range(n):
        qc.cx(qubit_value, n)

    qc.barrier()

    # Place X gates
    for qubit, qubit_value in enumerate(bit_str):
        if qubit_value == '1':
            qc.x(qubit)

    print('Output:', bit_str)
    if draw:
        draw_quantum_circuit(qc, draw_bloch_sphere=False)

    return qc


def deutsch_jozsa():
    n = 3
    dj_circuit = QuantumCircuit(n + 1, n)

    # Apply H-gates
    for qubit in range(n):
        dj_circuit.h(qubit)

    # Put last qubit in state |->
    dj_circuit.x(n)
    dj_circuit.h(n)

    # Apply an oracle
    # dj_circuit = dj_circuit.compose(balanced_oracle())
    dj_circuit = dj_circuit.compose(constant_oracle())

    # Repeat H-gates
    for qubit in range(n):
        dj_circuit.h(qubit)

    dj_circuit.barrier()

    # Measure
    for i in range(n):
        dj_circuit.measure(i, i)

    draw_quantum_circuit(dj_circuit, draw_final_state=False, draw_bloch_sphere=False, draw_histogram=True)


def dj_oracle(case: str, n: int) -> Gate:
    # We need to make a QuantumCircuit object to return
    # This circuit has n+1 qubits: the size of the input,
    # plus one output qubit
    oracle_qc = QuantumCircuit(n + 1)

    # First, let's deal with the case in which oracle is balanced
    if case == 'balanced':
        # First generate a random number that tells us which CNOTs to
        # wrap in X-gates:
        b = np.random.randint(1, 2 ** n)

        # Next, format 'b' as a binary string of length 'n', padded with zeros:
        b_str = format(b, '0' + str(n) + 'b')
        print(f'Oracle bit string: {b_str}')

        # Next, we place the first X-gates. Each digit in our binary string
        # corresponds to a qubit, if the digit is 0, we do nothing, if it's 1
        # we apply an X-gate to that qubit:
        for qubit in range(len(b_str)):
            if b_str[qubit] == '1':
                oracle_qc.x(qubit)

        # Do the controlled-NOT gates for each qubit, using the output qubit
        # as the target:
        for qubit in range(n):
            oracle_qc.cx(qubit, n)

        # Next, place the final X-gates
        for qubit in range(len(b_str)):
            if b_str[qubit] == '1':
                oracle_qc.x(qubit)

    # Case in which oracle is constant
    if case == 'constant':
        # First decide what the fixed output of the oracle will be
        # (either always 0 or always 1)
        output = np.random.randint(2)
        if output == 1:
            oracle_qc.x(n)

    oracle_gate = oracle_qc.to_gate()
    oracle_gate.name = "Oracle"  # To show when we display the circuit
    return oracle_gate


def dj_algorithm(oracle: Gate, n: int):
    dj_circuit = QuantumCircuit(n + 1, n)

    # Set up the output qubit:
    dj_circuit.x(n)
    dj_circuit.h(n)

    # And set up the input register:
    for qubit in range(n):
        dj_circuit.h(qubit)

    # Let's append the oracle gate to our circuit:
    dj_circuit.append(oracle, range(n + 1))

    # Finally, perform the H-gates again and measure:
    for qubit in range(n):
        dj_circuit.h(qubit)

    for i in range(n):
        dj_circuit.measure(i, i)

    return dj_circuit


def least_busy_backend(n):
    # Load our saved IBMQ accounts and get the least busy backend device with greater than or equal to (n+1) qubits
    # Note: the acconut is saved on disk (~/.qiskit) if you have run save_account() once
    IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q')
    backend = least_busy(provider.backends(filters=lambda x: x.configuration().n_qubits >= (n + 1) and
                                                             not x.configuration().simulator and
                                                             x.status().operational == True))
    print('least busy backend: ', backend)
    return backend


# Run our circuit on the least busy backend. Monitor the execution of the job in the queue

def run_quantum_job(backend: BaseBackend, circuit: QuantumCircuit):
    shots = 1024
    job = execute(circuit, backend=backend, shots=shots, optimization_level=3)
    job_monitor(job, interval=2)

    # Get the results of the computation
    results = job.result()
    answer = results.get_counts()
    plot_histogram(answer).show()


def execute_dj():
    n = 4
    oracle = dj_oracle('balanced', n)
    dj_circuit = dj_algorithm(oracle, n)
    draw_quantum_circuit(dj_circuit, draw_final_state=False, draw_bloch_sphere=False, draw_histogram=True)
    backend = least_busy_backend(n)
    run_quantum_job(backend, dj_circuit)


def solve_textbook_problems():
    n = 4
    for i in (0, 1, 2, 3, 4):
        oracle = problems.dj_problem_oracle(i)
        dj_circuit = dj_algorithm(oracle, n)
        backend = Aer.get_backend('statevector_simulator')
        results = execute(dj_circuit, backend).result().get_counts()
        score = 0
        for key, value in results.items():
            if key.startswith('000'):
                score += value
        if score == 0:
            print(f'Oracle {i} is balanced: {results}')
        elif score == 1:
            print(f'Oracle {i} is constant: {results}')
        else:
            print(f'Oracle {i} is neither balanced nor constant: {results}')


if __name__ == '__main__':
    # step_1()
    # make_oracle_balanced()
    # constant_oracle(draw=True)
    # balanced_oracle(draw=True)
    # deutsch_jozsa()
    # execute_dj()
    solve_textbook_problems()
